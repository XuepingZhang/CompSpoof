import sys
import os

import numpy as np

import logging
def calculate_EER(cm_scores_file):
    # Replace CM scores with your own scores or provide score file as the
    # first argument.
    # cm_scores_file =  'score_cm.txt'
    # Replace ASV scores with organizers' scores or provide score file as
    # the second argument.
    # asv_score_file = 'ASVspoof2019.LA.asv.eval.gi.trl.scores.txt'

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv':
        10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }



    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)


    cm_scores = cm_data[:, 1].astype(np.float16)
    cm_keys = cm_data[:, 2]

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to
    # EER threshold
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    return eer_cm * 100



from sklearn.metrics import det_curve
import numpy as np
import logging



def compute_dcf(bonafide_scores, spoof_scores, threshold, C_miss=1.0, C_fa=10.0, P_spoof=0.05):
    # miss: 真实 bonafide 被判为 spoof（s < threshold）
    P_miss = np.mean(bonafide_scores < threshold)
    # fa: 伪造 spoof 被判为 bonafide（s >= threshold）
    P_fa = np.mean(spoof_scores >= threshold)
    DCF = C_miss * (1 - P_spoof) * P_miss + C_fa * P_spoof * P_fa
    return DCF

def compute_min_dcf(bonafide_scores, spoof_scores, C_miss=1.0, C_fa=10.0, P_spoof=0.05):
    # 遍历所有阈值求最小 DCF
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(np.unique(all_scores))
    dcf_list = [compute_dcf(bonafide_scores, spoof_scores, t, C_miss, C_fa, P_spoof) for t in thresholds]
    return min(dcf_list)

def calculate_EER_DCF_eval(score, labels):
    scores = score[:, 1].cpu().numpy().ravel()
    bona_scores = scores[labels == 0]
    all_spoof_scores = scores[labels == 1]

    overall_eer, eer_thr = compute_eer(bona_scores, all_spoof_scores)
    overall_min_dcf = compute_min_dcf(bona_scores, all_spoof_scores)
    overall_act_dcf = compute_dcf(bona_scores, all_spoof_scores, eer_thr)
    return overall_eer * 100, overall_min_dcf, overall_act_dcf

def calculate_EER_evel(cm_scores_file, output_file, unseen_types):
    # 读取数据
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    paths = cm_data[:, 0]
    scores = cm_data[:, 1].astype(np.float32)
    labels = cm_data[:, 2]

    bona_scores = scores[labels == 'bonafide']
    all_spoof_scores = scores[labels == 'spoof']

    eer_dict = {}
    seen_spoof_scores = []
    unseen_spoof_scores = []

    with open(output_file, 'w') as f:

        attack_types = sorted(set([p.split('/')[1] for p in paths]))
        for attack in attack_types:
            attack_indices = [
                i for i, p in enumerate(paths)
                if f"/{attack}/" in p and labels[i] == 'spoof'
            ]
            spoof_scores = scores[attack_indices]
            if len(spoof_scores) == 0:
                continue

            eer = compute_eer(bona_scores, spoof_scores)[0] * 100
            eer_dict[attack] = eer

            if attack in unseen_types:
                unseen_spoof_scores.extend(spoof_scores)
            else:
                seen_spoof_scores.extend(spoof_scores)

            line = f"{attack:15s}: {eer:.2f}%"
            logging.info(line)
            f.write(line + "\n")

        seen_eer = compute_eer(bona_scores, np.array(seen_spoof_scores))[0] * 100
        unseen_eer = compute_eer(bona_scores, np.array(unseen_spoof_scores))[0] * 100
        overall_eer = compute_eer(bona_scores, all_spoof_scores)[0] * 100

        f.write("\n>>> Summary:\n")
        f.write(f"Seen   EER   : {seen_eer:.2f}%\n")
        f.write(f"Unseen EER   : {unseen_eer:.2f}%\n")
        f.write(f"Overall EER  : {overall_eer:.2f}%\n")


        logging.info(f">>> Seen   EER   : {seen_eer:.2f}%")
        logging.info(f">>> Unseen EER   : {unseen_eer:.2f}%")
        logging.info(f">>> Overall EER  : {overall_eer:.2f}%")

    return eer_dict, seen_eer, unseen_eer, overall_eer



def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, Les Sables d'Olonne,
          France, June 2018 (https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """

    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        logging.info('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit(
            'ERROR: Your prior probabilities should be positive and sum up to one.'
        )

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit(
            'ERROR: you should provide miss rate of spoof tests against your ASV system.'
        )

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit(
            'ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(
        bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
        cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit(
            'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?'
        )

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        logging.info('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(
            bonafide_score_cm.size, spoof_score_cm.size))
        logging.info('t-DCF MODEL')
        logging.info('   Ptar         = {:8.5f} (Prior probability of target user)'.
              format(cost_model['Ptar']))
        logging.info(
            '   Pnon         = {:8.5f} (Prior probability of nontarget user)'.
            format(cost_model['Pnon']))
        logging.info(
            '   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.
            format(cost_model['Pspoof']))
        logging.info(
            '   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'
            .format(cost_model['Cfa_asv']))
        logging.info(
            '   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'
            .format(cost_model['Cmiss_asv']))
        logging.info(
            '   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'
            .format(cost_model['Cfa_cm']))
        logging.info(
            '   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'
            .format(cost_model['Cmiss_cm']))
        logging.info(
            '\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)'
        )

        if C2 == np.minimum(C1, C2):
            logging.info(
                '   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(
                    C1 / C2))
        else:
            logging.info(
                '   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(
                    C2 / C1))

    return tDCF_norm, CM_thresholds



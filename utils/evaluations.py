def report_prf(tp, fp, fn, phase, logger=None, return_dict=False):
    # For the detection Precision, Recall and F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    if phase and logger:
        logger.info(f"The {phase} result is: "
                    f"{precision:.4f}/{recall:.4f}/{f1_score:.4f} -->\n"
                    # f"precision={precision:.6f}, recall={recall:.6f} and F1={f1_score:.6f}\n"
                    f"support: TP={tp}, FP={fp}, FN={fn}")
    if return_dict:
        ret_dict = {
            f'{phase}_p': precision,
            f'{phase}_r': recall,
            f'{phase}_f1': f1_score}
        return ret_dict
    return precision, recall, f1_score
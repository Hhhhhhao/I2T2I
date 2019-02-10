from eval_metrics.bleu.bleu import Bleu
from eval_metrics.meteor.meteor import Meteor
from eval_metrics.rouge.rouge import Rouge
from eval_metrics.cider.cider import Cider
from eval_metrics.spice.spice import Spice


def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)

    for i, sc in enumerate(score):
        print('belu_{} = {:.4f}'.format(i+1, sc))

    # return bleu_1, bleu_2, bleu_3, bleu_4
    return score[0], score[1], score[2], score[3]


def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    print('cider = %s' % score)
    return score


def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    print('meter = %s' % score)
    return score


def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    print('rouge = %s' % score)
    return score


def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print('spice = %s' % score)
    return score


def compute_score(gts, res):
    bleu_1, bleu_2, bleu_3, bleu_4 = bleu(gts, res)
    cider_score = cider(gts, res)
    meteor_score = meteor(gts, res)
    rouge_score = rouge(gts, res)
    spice_score = spice(gts, res)

    metric_names = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
    metric_values = [bleu_1, bleu_2, bleu_3, bleu_4, meteor_score, rouge_score, cider_score, spice_score]
    results = {}

    for name, score in zip(metric_names, metric_values):
        results[name] = score

    return results
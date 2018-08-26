import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np
import argparse


def significance_test(congruent, incongruent):
    return scipy.stats.wilcoxon(congruent, incongruent)


def calculate_awareness(congruent, incongruent, is_logprob=False):
    '''
    Calculates Equation 1.
    '''
    deltas = []
    awarenesses = []
    total_awareness = 0.
    for c, i in zip(congruent, incongruent):
        if is_logprob:
            delta = i - c # Equation 2 with reversed operands
        else:
            delta = c - i  # Equation 2.
        deltas.append(delta)
        a = awareness_function(delta)
        total_awareness += a
        awarenesses.append(a)

    deltas = np.array(deltas)
    total_awareness /= len(congruent)
    return deltas, total_awareness


def awareness_function(delta):
    '''
    Implements Eq (1).
    We simply return the delta_value, this models the
    awareness of a model as a linear relationship: a(x) = x.
    '''
    return delta


def clean_data(data):
    '''
    Transforms all of the data into floats instead of strings.
    '''
    return [float(x) for x in data.readlines()]


def main(args):
    '''
    Loads the scores from the congruent and incongruent text files.
    Cleans the scores data, as necessary.
    '''
    congruent = clean_data(args.congruent)
    incongruents = []
    deltas = []
    awarenesses = []
    ps = []
    scores = []
    for i in args.incongruent:
        incongruent = clean_data(i)
        mean_score = np.mean(incongruent)
        scores.append(mean_score)
        delta, mean_awareness = calculate_awareness(congruent,
                                                     incongruent,
                                                     args.logprob)
        stat, p = significance_test(congruent, incongruent)
        print('Mean awareness: {:.5f}, T={}, p={:.3f}'.format(mean_awareness,
                                                              stat, p))
        deltas.append(delta)
        awarenesses.append(mean_awareness)
        ps.append(p)

    # combine the different p-values using Fisher's method
    stat, p = scipy.stats.combine_pvalues(ps, method='fisher')
    print("Mean congruent score: {}".format(np.mean(congruent)))
    print("Mean incongruent score: {} +- {}".format(np.mean(scores), np.std(scores)))
    print("Fisher's method: Chi-Squared={}, p={:.7f}".format(stat, p))
    print("Avearge awareness: {} +- {}".format(np.mean(awarenesses), np.std(awarenesses)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congruent",
                        help="Path to file containing the scores for the model\
                              with the congruent visual context",
                        type=argparse.FileType('r'),
                        required=True)
    parser.add_argument("--incongruent",
                        help="Path to files containing the scores for the model\
                              with the incongruent visual contexts",
                        type=argparse.FileType('r'), nargs="+",
                        required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logprob", action="store_true")
    group.add_argument("--meteor", action="store_true")
    main(parser.parse_args())

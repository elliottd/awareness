import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np
import argparse


def plot_histograms(deltas):
    '''
    Plot a histogram of the a_M values calculcated for each data point
    in the evaluation set.

    The data points that assign a better score in the incongruent visual
    context is coloured in red.
    '''
    mask1 = deltas > 0.
    mask2 = deltas <= 0.
    ax1 = sns.distplot(deltas[mask1], kde=False, hist=True, color='gray')
    ax2 = sns.distplot(deltas[mask2], kde=False, hist=True, color='red')
    ax1.tick_params(labelsize=15)
    plt.yscale('log')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    plt.show()


def significance_test(congruent, incongruent):
    return scipy.stats.wilcoxon(congruent, incongruent)


def calculate_awareness(congruent, incongruent, is_logprob=False):
    '''
    Calculates Eq. 2
    '''
    deltas = []
    awarenesses = []
    total_awareness = 0.
    for c, i in zip(congruent, incongruent):
        if is_logprob:
            delta = i - c
        else:
            delta = c - i
        deltas.append(delta)
        a = awareness_function(delta)  # Equation 1.
        total_awareness += a
        awarenesses.append(a)

    deltas = np.array(deltas)
    total_awareness /= len(congruent)
    return deltas, total_awareness


def awareness_function(p_value):
    '''
    Implements Eq (1).
    We simply return the delta_p_value, this models the
    awareness of a model as a linear relationship: a(x) = x.
    '''
    return p_value


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
    print("Mean congruent score: {}".format(np.mean(congruent)))
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
    print("Fisher's method: Chi-Squared={}, p={:.7f}".format(stat, p))
    print("Avearge awareness: {} +- {}".format(np.mean(awarenesses), np.std(awarenesses)))
    print("Average score: {} +- {}".format(np.mean(scores), np.std(scores)))
    if not args.no_plot:
        plot_histograms(deltas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--congruent",
                        help="Path to file containing the scores for the model\
                              with the congruent visual context",
                        type=argparse.FileType('r'))
    parser.add_argument("--incongruent",
                        help="Path to files containing the scores for the model\
                              with the incongruent visual contexts",
                        type=argparse.FileType('r'), nargs="+")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logprob", action="store_true")
    group.add_argument("--meteor", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    main(parser.parse_args())

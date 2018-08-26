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

def package_awareness_numbers(congruent, incongruent, args):
    congruent = clean_data(congruent)
    incongruents = []
    deltas = []
    awarenesses = []
    ps = []
    scores = []
    for i in incongruent:
        incongruent = clean_data(i)
        mean_score = np.mean(incongruent)
        scores.append(mean_score)
        delta, mean_awareness = calculate_awareness(congruent,
                                                     incongruent,
                                                     args.logprob)
        deltas.append(delta)
        awarenesses.append(mean_awareness)
    plotdata = np.zeros(1014)
    for d in deltas:
        plotdata = np.concatenate((plotdata, d))
    plotdata = plotdata[1014:]  # strip off the leading zeros
    return plotdata


def main(args):
    '''
    Loads the scores from the congruent and incongruent text files.
    Cleans the scores data, as necessary.
    '''

    m1_congruent = args.model1[0]
    m1_incongruent = args.model1[1:]
    model1 = package_awareness_numbers(m1_congruent, m1_incongruent, args)

    m2_congruent = args.model2[0]
    m2_incongruent = args.model2[1:]
    model2 = package_awareness_numbers(m2_congruent, m2_incongruent, args)


    m3_congruent = args.model3[0]
    m3_incongruent = args.model3[1:]
    model3 = package_awareness_numbers(m3_congruent, m3_incongruent, args)

    sns.set_context("paper", rc={"grid.linewidth": 0.6, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelsize": 12})
    plot = sns.violinplot(data=[model1, model2, model3])
    plot.set(xticklabels=['trgmul', 'decinit', 'hierattn'], ylabel='Difference in Meteor score')


    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1",
                        help="Path to file containing the scores for the model\
                              with the congruent visual context",
                        type=argparse.FileType('r'), nargs="+",
                        required=True)
    parser.add_argument("--model2",
                        help="Path to files containing the scores for the model\
                              with the incongruent visual contexts",
                        type=argparse.FileType('r'), nargs="+",
                        required=True)
    parser.add_argument("--model3",
                        help="Path to files containing the scores for the model\
                              with the incongruent visual contexts",
                        type=argparse.FileType('r'), nargs="+",
                        required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--logprob", action="store_true")
    group.add_argument("--meteor", action="store_true")
    main(parser.parse_args())

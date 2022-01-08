from argparse import ArgumentParser
from difflib import SequenceMatcher
from enum import Enum
from os import path, makedirs, getcwd
from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')
Gender = Enum('Gender', ['Male', 'Female', 'Other'])
Experience = Enum('Experience', ['one_or_less', 'two_to_four', 'five_or_more'])
Group = Enum('Group', ['Meaningful', 'Meaningless'])
Age = Enum('Age', ['age_20_24', 'age_25_29', 'age_30_34', 'age_35_plus', ])

RESULTS_DIR = f"results_{strftime('%y%m%d_%H%M%S', localtime())}"
ANSWER_KEY = \
    {5: {Group.Meaningful: 'multiply_digits',
         Group.Meaningless: 'digits_calc'},
     2: {Group.Meaningful: 'indices_sum_to_target',
         Group.Meaningless: 'find_right_numbers'},
     1: {Group.Meaningful: 'words_counter',
         Group.Meaningless: 'words_process'},
     3: {Group.Meaningful: 'sum_square_diff',
         Group.Meaningless: 'calc_diff_alg'},
     4: {Group.Meaningful: 'sum_exp_digits',
         Group.Meaningless: 'compute_exp_sum'},
     }


def preprocess_data(filename: str):

    global df

    if path.exists(f"{filename[:-4]}.pkl"):
        df = pd.read_pickle(f"{filename[:-4]}.pkl")

    else:
        assert filename[-4:] == '.csv'
        df = pd.read_csv(filename)

        df = df.drop(labels='Timestamp', axis=1)
        df = df.drop(labels='Do you agree?', axis=1)
        df = df.drop(labels='According to the text, why do penguins waddle?', axis=1)

        df.columns = ['age', 'gender', 'experience', 'group',
                      'a1', 'a2', 'a3', 'a4', 'a5']

        df['experience'] = df['experience'].apply(
            lambda x:
            Experience.one_or_less if x == '0-1 years' else
            Experience.two_to_four if x == '2-4 years' else
            Experience.five_or_more)

        df['gender'] = df['gender'].apply(
            lambda x:
            Gender.Male if x == 'Male' else
            Gender.Female if x == 'Female' else
            Gender.Other)

        df['age'] = df['age'].apply(
            lambda x:
            Age.age_20_24 if 20 <= int(x) < 25 else
            Age.age_25_29 if 25 <= int(x) < 30 else
            Age.age_30_34 if 30 <= int(x) < 35 else
            Age.age_35_plus)

        df['group'] = df['group'].apply(
            lambda x:
            Group.Meaningful if int(x) % 2 != 0
            else Group.Meaningless)

        for a in ['a1', 'a2', 'a3', 'a4', 'a5']:
            df[a + '_score'] = df.apply(
                lambda row:
                ((SequenceMatcher(
                    None, row[a].lower().replace(' ', '_'),
                    ANSWER_KEY[int(a[-1])][row['group']]).ratio()) ** 2),
                axis=1)
            df

        df.to_pickle(f"{filename[:-4]}.pkl")


def plot_by_q(str_dist_th: float = 1.0):
    plt.clf()

    labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    meaningful_success, meaningless_success = [], []
    meaningful_df = df[df['group'] == Group.Meaningful]
    meaningless_df = df[df['group'] == Group.Meaningless]
    for a in ['a1', 'a2', 'a3', 'a4', 'a5']:
        meaningful_success.append(
            round(100 * meaningful_df[
                meaningful_df[a + '_score'] >= str_dist_th].size /
                  meaningful_df.size, 1))
        meaningless_success.append(
            round(100 * meaningless_df[
                meaningless_df[a + '_score'] >= str_dist_th].size /
                  meaningless_df.size, 1))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, meaningful_success, width,
           label='Meaningful')
    ax.bar(x + width / 2, meaningless_success, width,
           label='Meaningless')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Success rate [%]')
    ax.set_ylim((0, 40))
    ax.set_title(f"Scores by question and group (th={str_dist_th})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        plt.savefig(path.join(RESULTS_DIR, f"plot_by_q_{str_dist_th}.pdf"))


def plot_by_enum(enum_type_name: str, enum_type: Enum):
    plt.clf()

    str_dist_th: float = 1.0

    labels = [enum_type[k] for k in list(enum_type.__members__)]
    label_type_name = type(labels[0]).__name__.lower()
    meaningful_success, meaningless_success = [], []

    for label in labels:
        label_df = df[df[label_type_name] == label]
        meaningful_df = label_df[label_df['group'] == Group.Meaningful]
        meaningless_df = label_df[label_df['group'] == Group.Meaningless]

        if len(meaningful_df) > 0:
            meaningful_success.append(round((100 * meaningful_df[
                ['a1_score',
                 'a2_score',
                 'a3_score',
                 'a4_score',
                 'a5_score']].sum().sum() / meaningful_df.size), 1))
        else:
            meaningful_success.append(0)

        if len(meaningless_df) > 0:
            meaningless_success.append(round((100 * meaningless_df[
                ['a1_score',
                 'a2_score',
                 'a3_score',
                 'a4_score',
                 'a5_score']].sum().sum() / meaningless_df.size), 1))
        else:
            meaningless_df.append(0)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, meaningful_success, width,
           label='Meaningful')
    ax.bar(x + width / 2, meaningless_success, width,
           label='Meaningless')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Success rate [%]')
    ax.set_ylim((0, round(max(max(meaningful_success),
                              max(meaningless_success)), -1) + 10))
    ax.set_title(f"Scores by {label_type_name} and group (th={str_dist_th})")
    ax.set_xticks(x)
    ax.set_xticklabels([label.name for label in labels])
    ax.legend(loc='upper left')

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        plt.savefig(path.join(RESULTS_DIR, f"plot_by_enum_{enum_type_name}.pdf"))


def plot_responders_data(enum_type_name: str, enum_type: Enum):
    plt.clf()

    labels = [enum_type[k] for k in list(enum_type.__members__)]
    label_type_name = type(labels[0]).__name__.lower()
    counter = []

    for label in labels:
        label_df = df[df[label_type_name] == label]
        counter.append(len(label_df) / 72 * 100)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x, counter, width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Responders rate [%]')
    ax.set_ylim((0, 80))
    ax.set_title(f"Responders by {enum_type_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([label.name for label in labels])

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        plt.savefig(path.join(RESULTS_DIR, f"plot_responders_data_{enum_type_name}.pdf"))


def plot_success_histogram(str_dist_th: float = 1.0):
    plt.clf()

    meaningful_success, meaningless_success = np.zeros(6), np.zeros(6)
    meaningful_df = df[df['group'] == Group.Meaningful]
    meaningless_df = df[df['group'] == Group.Meaningless]

    meaningful_value_counts = (meaningful_df[[
        'a1_score', 'a2_score', 'a3_score', 'a4_score', 'a5_score']]
                               >= str_dist_th).sum(axis=1).value_counts()
    meaningless_value_counts = (meaningless_df[[
        'a1_score', 'a2_score', 'a3_score', 'a4_score', 'a5_score']]
                                >= str_dist_th).sum(axis=1).value_counts()

    for k, v in enumerate(list(meaningful_value_counts)):
        meaningful_success[k] = v / len(meaningful_df) * 100
    for k, v in enumerate(list(meaningless_value_counts)):
        meaningless_success[k] = v / len(meaningful_df) * 100

    fig, ax = plt.subplots()
    ax.plot(meaningful_success, marker='.', label='Meaningful')
    ax.plot(meaningless_success, marker='.', label='Meaningless')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Correct answers [count]')
    ax.set_ylabel('Respondents [%]')
    ax.set_ylim((0, 75))
    ax.set_title(f"Amount of correct answers per respondent (th={str_dist_th})")
    ax.legend(loc='upper right')

    fig.tight_layout()
    if args.show:
        plt.show(legend=None)
    else:
        plt.savefig(path.join(RESULTS_DIR, f"plot_success_histogram_{str_dist_th}.pdf"))


def plot_success_cdf():
    meaningful_df = df[df['group'] == Group.Meaningful]
    meaningless_df = df[df['group'] == Group.Meaningless]

    for n in range(1, 5 + 1):
        column_name = f"a{n}_score"
        plt.clf()
        fig, ax = plt.subplots()

        meaningful_stats_df = meaningful_df.groupby(column_name)[column_name].agg('count').pipe(pd.DataFrame) \
            .rename(columns={column_name: 'frequency'})
        meaningless_stats_df = meaningless_df.groupby(column_name)[column_name].agg('count').pipe(pd.DataFrame) \
            .rename(columns={column_name: 'frequency'})
        meaningful_stats_df['pdf'] = meaningful_stats_df['frequency'] / sum(meaningful_stats_df['frequency'])
        meaningless_stats_df['pdf'] = meaningless_stats_df['frequency'] / sum(meaningless_stats_df['frequency'])
        meaningful_stats_df['cdf'] = meaningful_stats_df['pdf'].cumsum()
        meaningless_stats_df['cdf'] = meaningless_stats_df['pdf'].cumsum()

        ax.plot(meaningful_stats_df['cdf'], drawstyle='steps-post', marker='.', label='Meaningful', clip_on=False)
        ax.plot(meaningless_stats_df['cdf'], drawstyle='steps-post', marker='.', label='Meaningless', clip_on=False)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Score')
        ax.set_xlim((0, 1))
        ax.set_ylabel('Probability')
        ax.set_ylim((0, 1))
        ax.set_title(f"CDF of answer score for Q{n}")
        ax.legend(loc='upper left')
        ax.spines[:].set_visible(False)

        fig.tight_layout()
        if args.show:
            plt.show(legend=None)
        else:
            plt.savefig(path.join(RESULTS_DIR, f"plot_success_cdf_q{n}.pdf"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, default='survey_results_2110.csv', help='File name of CSV data source')
    parser.add_argument('--show', type=bool, default=False, help='If True, results will be displayed. Else, saved.')
    args = parser.parse_args()

    if not args.show:
        current_directory = getcwd()
        final_directory = path.join(current_directory, RESULTS_DIR)
        if not path.exists(final_directory):
            makedirs(final_directory)

    preprocess_data(filename=args.filename)

    plot_by_q(str_dist_th=0.5)
    plot_by_q(str_dist_th=0.8)
    plot_by_q(str_dist_th=1.0)
    plot_by_enum(enum_type_name='Gender', enum_type=Gender)
    plot_by_enum(enum_type_name='Experience', enum_type=Experience)
    plot_by_enum(enum_type_name='Age', enum_type=Age)
    plot_responders_data(enum_type_name='Gender', enum_type=Gender)
    plot_responders_data(enum_type_name='Experience', enum_type=Experience)
    plot_responders_data(enum_type_name='Age', enum_type=Age)
    plot_success_histogram(str_dist_th=0.5)
    plot_success_histogram(str_dist_th=0.8)
    plot_success_histogram(str_dist_th=1.0)
    plot_success_cdf()

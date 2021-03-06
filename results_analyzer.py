from argparse import ArgumentParser
from difflib import SequenceMatcher
from enum import Enum
from os import path, makedirs, getcwd
from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Font size options: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
plt.rcParams.update({
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'legend.fontsize': 'large',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium',
})
plt.style.use('ggplot')

Gender = Enum('Gender', ['Male', 'Female', 'Other'])
Experience = Enum('Experience', ['one_or_less', 'two_to_four', 'five_or_more'])
Group = Enum('Group', ['Meaningful', 'Meaningless'])
Age = Enum('Age', ['age_20_24', 'age_25_29', 'age_30_34', 'age_35_plus', ])

TIMESTAMP = f"results_{strftime('%y%m%d_%H%M%S', localtime())}"

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

COLUMN_ORDER_DROP_NAME = {
    "basic": "Reordered",
    "reorder": "Corrected"
}

COLUMNS_DROPPED = [
    'Timestamp',
    'Do you agree?',
    'Answer 1',
    'Answer 2',
    'Answer 3',
    'Answer 4',
    'Answer 5',
    'Valid 1',
    'Valid 2',
    'Valid 3',
    'Valid 4',
    'Valid 5',
    'According to the text, why do penguins waddle?',
]


def filter_valid_answers_by_label(sub_df: pd.DataFrame, a_len: np.ndarray):
    """ Fixes to score 0 if label is not 1 and counts into a_len when label is 1"""
    for j in range(5):
        for i, tag in enumerate(sub_df[f"l{j + 1}"]):  # iterate over row s
            c_i = sub_df.columns.get_loc(f"a{j + 1}_score")
            if str(tag) != "1":
                sub_df.iloc[i, c_i] = 0
            else:
                a_len[j] += 1


def _score(str1: str, str2: str):
    return max(SequenceMatcher(a=str1, b=str2).ratio(),
               SequenceMatcher(a=str2, b=str1).ratio())


def preprocess_data(filename: str, order_str: str):
    global a_len_meaningful
    a_len_meaningful = np.zeros(5)
    global a_len_meaningless
    a_len_meaningless = np.zeros(5)
    global a_len_general
    a_len_general = np.zeros(5)
    global df

    if path.exists(f"{filename[:-4]}_processed_{order_str}.pkl"):
        df = pd.read_pickle(f"{filename[:-4]}_processed_{order_str}.pkl")

    else:

        assert filename[-4:] == '.csv'
        df = pd.read_csv(filename)

        for column_name in df.columns:
            if COLUMN_ORDER_DROP_NAME[order_str] in column_name:
                df = df.drop(labels=column_name, axis=1)

        for column_name in COLUMNS_DROPPED:
            df = df.drop(labels=column_name, axis=1)

        df.columns = ['age', 'gender', 'experience', 'group',
                      'a1', 'l1',
                      'a2', 'l2',
                      'a3', 'l3',
                      'a4', 'l4',
                      'a5', 'l5'
                      ]

        df.astype(dtype=str)

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

        for i in range(1, 5 + 1):
            answer_col = f"a{i}"

            df[answer_col + '_score'] = df.apply(
                func=lambda row:
                _score(str1=row[answer_col].lower().replace(' ', '_'), str2=ANSWER_KEY[int(i)][row['group']]),
                axis=1)

    meaningful_df = df[df['group'] == Group.Meaningful]
    meaningless_df = df[df['group'] == Group.Meaningless]

    # Calculate validity of each occurrence of 1 for each group and in general (the general is to correct all scores)
    filter_valid_answers_by_label(meaningful_df, a_len_meaningful)
    filter_valid_answers_by_label(meaningless_df, a_len_meaningless)
    filter_valid_answers_by_label(df, a_len_general)

    df.to_pickle(f"{filename[:-4]}_processed_{order_str}.pkl")
    df.to_csv(f"{filename[:-4]}_processed_{order_str}.csv")


def plot_by_q(str_dist_th: float = 1.0):
    plt.clf()

    labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    meaningful_success, meaningless_success = [], []
    meaningful_df = df[df['group'] == Group.Meaningful]
    meaningless_df = df[df['group'] == Group.Meaningless]
    for i, a in enumerate(['a1', 'a2', 'a3', 'a4', 'a5']):
        meaningful_success.append(
            round(100 * meaningful_df[
                meaningful_df[a + '_score'] >= str_dist_th].shape[0] /  # changed to shape[0] from size
                  a_len_meaningful[i], 1))
        meaningless_success.append(
            round(100 * meaningless_df[
                meaningless_df[a + '_score'] >= str_dist_th].shape[0] /  # changed to shape[0] from size
                  a_len_meaningless[i], 1))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, meaningful_success, width,
           label='Meaningful')
    ax.bar(x + width / 2, meaningless_success, width,
           label='Meaningless')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Success rate [%]')
    ax.set_ylim((0, 80))
    ax.set_title(f"Scores by question and group (th={str_dist_th})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')

    fig.tight_layout()

    if args.show:
        plt.show()
    else:
        plt.savefig(path.join(output_dir, f"plot_by_q_{str_dist_th}.pdf"))


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
        plt.savefig(path.join(output_dir, f"plot_by_enum_{enum_type_name}.pdf"))


def plot_responders_data(enum_type_name: str, enum_type: Enum):
    plt.clf()

    labels = [enum_type[k] for k in list(enum_type.__members__)]
    label_type_name = type(labels[0]).__name__.lower()
    counter = []
    total_responders = df.shape[0]

    for label in labels:
        label_df = df[df[label_type_name] == label]
        counter.append(len(label_df) / total_responders * 100)

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
        plt.savefig(path.join(output_dir, f"plot_responders_data_{enum_type_name}.pdf"))


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
        plt.savefig(path.join(output_dir, f"plot_success_histogram_{str_dist_th}.pdf"))


def plot_success_cdf():
    meaningful_df = df[df['group'] == Group.Meaningful]
    meaningless_df = df[df['group'] == Group.Meaningless]

    for i in range(1, 5 + 1):
        plt.clf()
        fig, ax = plt.subplots()
        column_name = f"a{i}_score"

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
        ax.set_title(f"CDF of answer score for the function of '{ANSWER_KEY[i][Group.Meaningful]}'")
        ax.legend(loc='upper left')
        ax.spines[:].set_visible(False)

        fig.tight_layout()
        if args.show:
            plt.show(legend=None)
        else:
            plt.savefig(path.join(output_dir, f"plot_success_cdf_q{i}.pdf"))
        meaningful_stats_df.to_csv(path.join(output_dir, "cdf", f"Q{i}_meaningful.csv"))
        meaningless_stats_df.to_csv(path.join(output_dir, "cdf", f"Q{i}_meaningless.csv"))


def plot_success_cdf_order_comparison():
    for i in range(1, 5 + 1):
        plt.clf()
        fig, ax = plt.subplots()

        meaningful_basic_stats_df = pd.read_csv(f"{TIMESTAMP}/basic/cdf/Q{i}_meaningful.csv")
        meaningless_basic_stats_df = pd.read_csv(f"{TIMESTAMP}/basic/cdf/Q{i}_meaningless.csv")
        meaningful_reorder_stats_df = pd.read_csv(f"{TIMESTAMP}/reorder/cdf/Q{i}_meaningful.csv")
        meaningless_reorder_stats_df = pd.read_csv(f"{TIMESTAMP}/reorder/cdf/Q{i}_meaningless.csv")

        ax.plot(meaningful_basic_stats_df[f"a{i}_score"], meaningful_basic_stats_df['cdf'],
                drawstyle='steps-post', marker='.', label='Meaningful (Basic)', clip_on=False, ls='-', c='r')
        ax.plot(meaningless_basic_stats_df[f"a{i}_score"], meaningless_basic_stats_df['cdf'],
                drawstyle='steps-post', marker='.', label='Meaningless (Basic)', clip_on=False, ls='-', c='b')
        ax.plot(meaningful_reorder_stats_df[f"a{i}_score"], meaningful_reorder_stats_df['cdf'],
                drawstyle='steps-post', marker='.', label='Meaningful (Reorder)', clip_on=False, ls='--', c='r')
        ax.plot(meaningless_reorder_stats_df[f"a{i}_score"], meaningless_reorder_stats_df['cdf'],
                drawstyle='steps-post', marker='.', label='Meaningless (Reorder)', clip_on=False, ls='--', c='b')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('Score')
        ax.set_xlim((0, 1))
        ax.set_ylabel('Probability')
        ax.set_ylim((0, 1))
        ax.set_title(f"CDF of answer score for the function of '{ANSWER_KEY[i][Group.Meaningful]}'")
        ax.legend(loc='upper left')
        ax.spines[:].set_visible(False)

        fig.tight_layout()
        if args.show:
            plt.show(legend=None)
        else:
            plt.savefig(path.join(cwd, TIMESTAMP, f"plot_success_cdf_q{i}.pdf"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, default='survey_results_2112_reorder.csv',
                        help='File name of CSV data source')
    parser.add_argument('--show', type=bool, default=False, help='If True, results will be displayed. Else, saved.')
    args = parser.parse_args()

    global output_dir

    for order in COLUMN_ORDER_DROP_NAME.keys():
        if not args.show:
            cwd = getcwd()

            output_dir = path.join(cwd, TIMESTAMP, order)
            if not path.exists(output_dir):
                makedirs(output_dir)
            output_cdf_dir = path.join(output_dir, "cdf")
            if not path.exists(output_cdf_dir):
                makedirs(output_cdf_dir)

        preprocess_data(filename=args.filename, order_str=order)

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
    plot_success_cdf_order_comparison()

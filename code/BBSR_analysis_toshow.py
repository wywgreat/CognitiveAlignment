#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict
from sklearn.metrics import precision_score
from matplotlib.colors import LinearSegmentedColormap

def load_filter_concept_cat_dic(file_path, filter_concept_type_lst,
                                filter_concept_super_category_lst):

    df = pd.read_excel (file_path)

    concept_type_dict = {}
    concept_super_category_dict = {}
    concept_category_dict = {}


    for index, row in df.iterrows ():
        concept = row['Word']
        concept_type = row['Type']
        super_category = row['Super Category']
        category = row['Category']

        concept_type_dict[concept] = concept_type
        concept_super_category_dict[concept] = super_category
        # concept_category_dict[concept] = category

    filter_concept_type_dict = {k: v for k, v in concept_type_dict.items() if v in filter_concept_type_lst}
    filter_concept_super_category_dict = {k: v for k, v in concept_super_category_dict.items() if v in filter_concept_super_category_lst}
    #filter_concept_super_category_dict = {k: v for k, v in concept_category_dict.items() if k in filter_concept_category_lst}
    return filter_concept_type_dict, filter_concept_super_category_dict #, filter_concept_category_dict#



#################################################MultisensoryProfileAnalysis#################################################
def multisensory_profile_analysis(LLM_name, LLM_MEAN_file, multisensory_profile_folder, human_color, model_colors_dic):
    if not os.path.exists(multisensory_profile_folder):
        os.makedirs(multisensory_profile_folder)
    analysis_data_folder = '../results/BBSR_' + LLM_name + '/processed/'
    BBSR_file = "../data/[BBSR]WordSet1_Ratings.xlsx"
    dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()
    all65dims_lst = list(dim_domain_dic.keys())
    # load data
    human_data = (pd.read_excel(BBSR_file, usecols=['Word'] + all65dims_lst,
                                na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                  .rename(columns={'Word': 'Concept'})
                  .fillna(0)
                  .drop_duplicates()
                  .query("Concept != 'used'")
                  .set_index('Concept'))
    # print(human_data)
    LLM_data = (pd.read_excel(analysis_data_folder + LLM_MEAN_file, usecols=['Concept'] + all65dims_lst,
                              na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                .fillna(0)
                .drop_duplicates()
                .query("Concept != 'used'")
                .set_index('Concept'))


    plot_LLM_human_radar(human_data, LLM_data, all65dims_lst, LLM_name, multisensory_profile_folder, human_color, model_colors_dic)
def load_dim_domain_dic():
    dim_domain_dic = {
        "Vision": "Vision", "Bright": "Vision", "Dark": "Vision", "Color": "Vision", "Pattern": "Vision",
        "Large": "Vision", "Small": "Vision", "Motion": "Vision", "Biomotion": "Vision", "Fast": "Vision",
        "Slow": "Vision", "Shape": "Vision", "Complexity": "Vision", "Face": "Vision", "Body": "Vision",
        "Touch": "Somatic", "Temperature": "Somatic", "Texture": "Somatic", "Weight": "Somatic", "Pain": "Somatic",
        "Audition": "Audition", "Loud": "Audition", "Low": "Audition", "High": "Audition", "Sound": "Audition",
        "Music": "Audition", "Speech": "Audition", "Taste": "Gustation", "Smell": "Olfaction", "Head": "Motor",
        "UpperLimb": "Motor", "LowerLimb": "Motor", "Practice": "Motor", "Landmark": "Spatial", "Path": "Spatial",
        "Scene": "Spatial", "Near": "Spatial", "Toward": "Spatial", "Away": "Spatial", "Number": "Spatial",
        "Time": "Temporal", "Duration": "Temporal", "Long": "Temporal", "Short": "Temporal", "Caused": "Causal",
        "Consequential": "Causal", "Social": "Social", "Human": "Social", "Communication": "Social", "Self": "Social",
        "Cognition": "Cognition", "Benefit": "Emotion", "Harm": "Emotion", "Pleasant": "Emotion",
        "Unpleasant": "Emotion",
        "Happy": "Emotion", "Sad": "Emotion", "Angry": "Emotion", "Disgusted": "Emotion", "Fearful": "Emotion",
        "Surprised": "Emotion", "Drive": "Drive", "Needs": "Drive", "Attention": "Attention", "Arousal": "Attention"
    }
    domain_dimlst_dic = {
        "Vision": ["Vision", "Bright", "Dark", "Color", "Pattern", "Large", "Small", "Motion", "Biomotion", "Fast",
                   "Slow", "Shape", "Complexity", "Face", "Body"],
        "Somatic": ["Touch", "Temperature", "Texture", "Weight", "Pain"],
        "Audition": ["Audition", "Loud", "Low", "High", "Sound", "Music", "Speech"],
        "Gustation": ["Taste"],
        "Olfaction": ["Smell"],
        "Motor": ["Head", "UpperLimb", "LowerLimb", "Practice"],
        "Spatial": ["Landmark", "Path", "Scene", "Near", "Toward", "Away", "Number"],
        "Temporal": ["Time", "Duration", "Long", "Short"],
        "Causal": ["Caused", "Consequential"],
        "Social": ["Social", "Human", "Communication", "Self"],
        "Cognition": ["Cognition"],
        "Emotion": ["Benefit", "Harm", "Pleasant", "Unpleasant", "Happy", "Sad", "Angry", "Disgusted", "Fearful",
                    "Surprised"],
        "Drive": ["Drive", "Needs"],
        "Attention": ["Attention", "Arousal"]
    }
    return dim_domain_dic, domain_dimlst_dic
def plot_LLM_human_radar(human_data, LLM_data, dims_lst, LLM_name,multisensory_profile_folder, human_color, model_colors_dic):

    human_dims_mean_vec = [human_data[each_dim].mean() for each_dim in dims_lst]
    LLM_dims_mean_vec = [LLM_data[each_dim].mean() for each_dim in dims_lst]


    human_dims_mean_vec = np.array(human_dims_mean_vec)
    LLM_dims_mean_vec = np.array(LLM_dims_mean_vec)
    human_dims_mean_vec = np.concatenate((human_dims_mean_vec, [human_dims_mean_vec[0]]))
    LLM_dims_mean_vec = np.concatenate((LLM_dims_mean_vec, [LLM_dims_mean_vec[0]]))


    angles = np.linspace(0, 2 * np.pi, len(dims_lst), endpoint=False).tolist()
    angles += angles[:1]


    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))


    LLM_color = model_colors_dic.get(LLM_name, "#76B7B2")  # 默认颜色

    ax.fill(angles, human_dims_mean_vec, color=human_color, alpha=0.25, label='Human')
    ax.fill(angles, LLM_dims_mean_vec, color=LLM_color, alpha=0.25, label=LLM_name)

    ax.plot(angles, human_dims_mean_vec, color=human_color, linewidth=2)
    ax.plot(angles, LLM_dims_mean_vec, color=LLM_color, linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])

    for angle, label in zip(angles[:-1], dims_lst):
        rotation = np.degrees(angle)
        alignment = "right" if 90 < rotation < 270 else "left"
        ax.text(
            angle,
            ax.get_rmax() * 1.1,
            label,
            size=10,
            weight="bold",
            horizontalalignment=alignment,
            verticalalignment="center",
            rotation=rotation if alignment == "left" else rotation - 180,
        )


    ax.xaxis.set_tick_params(size=0, width=0)
    # ax.set_xticks([])
    ax.set_xticklabels([])

    # ax.legend(loc='best', bbox_to_anchor=(1.2, 1.2), fontsize=10)
    # plt.title(f'Radar Chart of Human and {LLM_name}', fontsize=14)

    plt.savefig(f"{multisensory_profile_folder}/BBSR_human_vs_{LLM_name}.png")



def load_rdm_concept_order(file_path, cattype, concept_cattype_order):
    df = pd.read_excel (file_path)
    concept_cattype_dict = {}

    for index, row in df.iterrows ():
        concept = row['Word']
        cat_type = row[cattype]
        concept_cattype_dict[concept] = cat_type
    concepts_in_cattype_order = []
    for cattype_inorder in concept_cattype_order:
        filter_concepts_lst = []
        for concept, each_cattype in concept_cattype_dict.items():
            if each_cattype == cattype_inorder:
                filter_concepts_lst.append(concept)
        concepts_in_cattype_order += filter_concepts_lst

    concepts_in_cattype_order_lst_without_used = [item for item in concepts_in_cattype_order if item != "used"]
    return concepts_in_cattype_order_lst_without_used


def rsa_analysis(LLM_name, LLM_MEAN_file, rsa_folder, cat_type, rdm_concept_cattype_based_order, concepts_in_cattype_order_lst):
    eachmodel_cattype_align_score_tuple_lst = []
    if not os.path.exists(rsa_folder):
        os.makedirs(rsa_folder)
    analysis_data_folder = '../results/BBSR_' + LLM_name + '/processed/'
    BBSR_file = "../data/[BBSR]WordSet1_Ratings.xlsx"
    dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()
    all65dims_lst = list(dim_domain_dic.keys())
    # load data
    human_data = (pd.read_excel(BBSR_file, usecols=['Word'] + all65dims_lst,
                                na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                  .rename(columns={'Word': 'Concept'})
                  .fillna(0)
                  .drop_duplicates()
                  .query("Concept != 'used'")
                  .set_index('Concept'))
    LLM_data = (pd.read_excel(analysis_data_folder + LLM_MEAN_file, usecols=['Concept'] + all65dims_lst,
                              na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                .fillna(0)
                .drop_duplicates()
                .query("Concept != 'used'")
                .set_index('Concept'))

    rdm_human = compute_rdm(human_data)
    rdm_llm = compute_rdm(LLM_data)
    concepts = human_data.index
    rdm_human_df = pd.DataFrame(rdm_human, index=concepts, columns=concepts)
    rdm_llm_df = pd.DataFrame(rdm_llm, index=concepts, columns=concepts)
    rdm_human_df = rdm_human_df.loc[concepts_in_cattype_order_lst, concepts_in_cattype_order_lst]
    rdm_llm_df = rdm_llm_df.loc[concepts_in_cattype_order_lst, concepts_in_cattype_order_lst]
    plot_rdm(rdm_human_df, "RDM in the Cognitive Space of Human", rsa_folder+"human_rdm.png")
    if LLM_name.endswith("-Instruct"):
        titlename = LLM_name.strip("-Instruct")
    else:
        if LLM_name == "gpt-3.5-turbo":
            titlename = "GPT-3.5 Turbo"
        if LLM_name == "gpt4omini":
            titlename = "GPT-4o mini"
    plot_rdm(rdm_llm_df, "RDM in the Cognitive Space of " + titlename, rsa_folder + LLM_name+ "_rdm.png")
    rdm_human_df.to_excel(rsa_folder + "human_rdm.xlsx")
    rdm_llm_df.to_excel(rsa_folder + LLM_name+ "human_rdm.xlsx")

    from scipy.stats import spearmanr
    human_rdm_vector = rdm_human[np.triu_indices_from(rdm_human, k=1)]
    llm_rdm_vector = rdm_llm[np.triu_indices_from(rdm_llm, k=1)]
    all_align_score, p_value = spearmanr(human_rdm_vector, llm_rdm_vector)
    eachmodel_cattype_align_score_tuple_lst.append((LLM_name, "ALL", all_align_score))

    for each_cattype in rdm_concept_cattype_based_order:
        concepts_in_cat_type_lst = load_rdm_concept_order(BBSR_file, cat_type, [each_cattype])

        filtered_human_data = human_data.loc[human_data.index.isin(concepts_in_cat_type_lst)]
        filtered_llm_data = LLM_data.loc[LLM_data.index.isin(concepts_in_cat_type_lst)]

        f_rdm_human = compute_rdm(filtered_human_data)
        f_rdm_llm = compute_rdm(filtered_llm_data)

        human_rdm_vector = f_rdm_human[np.triu_indices_from(f_rdm_human, k=1)]
        llm_rdm_vector = f_rdm_llm[np.triu_indices_from(f_rdm_llm, k=1)]

        eachcattype_align_score, p_value = spearmanr(human_rdm_vector, llm_rdm_vector)
        eachmodel_cattype_align_score_tuple_lst += [(LLM_name, each_cattype, eachcattype_align_score)]

    return eachmodel_cattype_align_score_tuple_lst

def compute_rdm(data):
    distances = pdist(data.values, metric='euclidean')
    rdm = squareform(distances)
    return rdm


def plot_rdm(rdm, title, filename):
    # my_colors = [(0.0, "#76a7c3"),  # 蓝色（位置 0）
    #          (0.5, "#fff0de"),  # 橙色（位置 0.5）
    #          (1.0, "#c18586")]  # 红色（位置 1）

    #my_cmap = LinearSegmentedColormap.from_list("custom_gradient", my_colors)
    plt.figure(figsize=(10, 8))
    sns.heatmap(rdm, cmap='coolwarm', square=True, cbar=True,vmin=0, vmax=33, # 为了统一色阶
                cbar_kws={'shrink': 0.8, 'aspect': 10})
    #“magma”
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def output_align_score_results(align_score_tuple_lst, output_path):
    df = pd.DataFrame(align_score_tuple_lst, columns=["LLM_name", "supercategory", "align_score"])

    df.set_index("LLM_name", inplace=True)

    df_pivot = df.pivot(columns="supercategory", values="align_score")

    excel_path = output_path
    df_pivot.to_excel(excel_path)

#################################################MultisensoryProfileAnalysis#################################################

#################################################ConsistencyAnalysis#################################################
def consistency_analysis(LLM_lst, corr_kl_mse_lst, focus_domain_lst, focus_colors, consistency_analysis_folder, filter_concept_type_lst,
                                filter_concept_super_category_lst, model_colors_dic, aoa_window_size_lst):
    if not os.path.exists(consistency_analysis_folder):
        os.makedirs(consistency_analysis_folder)
    for corr_kl_mse in corr_kl_mse_lst:
        # plot_allmoodels_corrklmse_barline(LLM_lst, corr_kl_mse, focus_domain_lst, focus_colors, consistency_analysis_folder)
        plot_typeconcepts_corrklmse(LLM_lst, corr_kl_mse, consistency_analysis_folder, filter_concept_type_lst,
                               filter_concept_super_category_lst)

        # plot_aoa_vs_corr_kl_mse(LLM_lst, consistency_analysis_folder + "AoA_corrklmse_lineplot/", corr_kl_mse,
        #                          model_colors_dic, aoa_window_size_lst)
        # plot_conc_vs_corr_kl_mse(LLM_lst, consistency_analysis_folder + "Conc_corrklmse_lineplot/", corr_kl_mse,
        #                          model_colors_dic, aoa_window_size_lst)


def plot_allmoodels_corrklmse_barline(LLM_lst, corr_kl_mse, focus_domain_lst, focus_colors, consistency_analysis_folder):
    plt.figure(figsize=(20, 12))
    num_models = len(LLM_lst)
    bar_width = 0.3 / num_models
    model_positions = np.arange(len(LLM_lst))

    linex_model_domainposition_dic = {}
    liney_model_domainmeans_dic = {}

    for LLM_name in LLM_lst:
        linex_model_domainposition_dic[LLM_name] = []
        liney_model_domainmeans_dic[LLM_name] = []
    for idx, domain in enumerate(focus_domain_lst):
        means = []
        for LLM_idx, LLM_name in enumerate(LLM_lst):
            corr_kl_mse_result_filepath = '../results/BBSR_' + LLM_name + '/statistics/micro_stats/' + \
                                          LLM_name + '_' + corr_kl_mse + '_micro_stats.xlsx'
            each_result_df = pd.read_excel(corr_kl_mse_result_filepath, sheet_name="Overall_Stats")
            each_result_df['Mean'] = each_result_df['Mean'].fillna(0)
            each_result_df['Standard Deviation'] = each_result_df['Standard Deviation'].fillna(0)
            sensory_meansd_dic = {row['Dimension']: (row['Mean'], row['Standard Deviation']) for _, row in
                                  each_result_df.iterrows()}

            key = domain + "_" + corr_kl_mse
            if key in sensory_meansd_dic:
                mean, std_dev = sensory_meansd_dic[key]
            else:
                mean, std_dev = 0, 0
            means.append(mean)
            all_positions = model_positions + idx * bar_width
            linex_model_domainposition_dic[LLM_name].append(all_positions[LLM_idx])
            liney_model_domainmeans_dic[LLM_name].append(mean)


        bar_positions = model_positions + idx * bar_width
        plt.bar(
            bar_positions,
            means,
            color=focus_colors[domain],
            edgecolor='black',
            width=bar_width,
            alpha=0.7,
            label=domain
        )

    output_columns = [i+"_"+corr_kl_mse for i in focus_domain_lst]
    df_output = pd.DataFrame(liney_model_domainmeans_dic, index=output_columns)
    df_output.to_excel(consistency_analysis_folder+"BBSRallmoodels_"+corr_kl_mse+".xlsx")

    labelname_lst = []
    for LLM_name in LLM_lst:
        plt.plot(
            linex_model_domainposition_dic[LLM_name],
            liney_model_domainmeans_dic[LLM_name],
            color="#1f78b4",
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=8
            )
        if LLM_name.endswith("-Instruct"):
            labelname = LLM_name.strip("-Instruct")
        else:
            labelname = LLM_name
        if labelname == "gpt-3.5-turbo":
            labelname = "GPT-3.5 Turbo"
        if labelname == "gpt4omini":
            labelname = "GPT-4o mini"
        labelname_lst.append(labelname)
    print(labelname_lst)
    plt.xticks(model_positions + bar_width * (len(focus_domain_lst) - 1) / 2, labelname_lst, fontsize=12) # x轴标签旋转角度#,rotation=45
    plt.xlabel('LLM Models', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    #plt.title(f'MSE Across LLMs by Dimension ({corr_kl_mse})', fontsize=16)
    plt.legend(loc='upper left',
            fontsize=16,
            title="Domains",
            title_fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(consistency_analysis_folder + f"BBSR_ConsistencyAnalysis_ALLmodels_{corr_kl_mse}.png")
def load_concept_cat_dic(file_path):
    df = pd.read_excel (file_path)

    concept_type_dict = {}
    concept_super_category_dict = {}
    concept_category_dict = {}

    for index, row in df.iterrows ():
        concept = row['Word']
        concept_type = row['Type']
        super_category = row['Super Category']
        category = row['Category']

        concept_type_dict[concept] = concept_type
        concept_super_category_dict[concept] = super_category
        concept_category_dict[concept] = category

    return concept_type_dict, concept_category_dict, concept_super_category_dict
def plot_typeconcepts_corrklmse(LLM_lst, corr_kl_mse, consistency_analysis_folder, filter_concept_type_lst,
                                filter_concept_super_category_lst):
    BBSR_file = "../data/[BBSR]WordSet1_Ratings.xlsx"
    concept_type_dict, concept_super_category_dict = load_filter_concept_cat_dic(BBSR_file,
                                                                            filter_concept_type_lst,
                                                                            filter_concept_super_category_lst)
    concept_super_category_dict
    # dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()
    # all65dims_lst = list(dim_domain_dic.keys())
    concept_type_data_list = []
    concept_super_category_data_list = []
    # concept_category_data_list = []
    for LLM_name in LLM_lst:
        # print(LLM_name)
        allDomain_corrklmse_file = '../results/BBSR_' + LLM_name + '/statistics/ALLDomain_corr_kl_mse.xlsx'
        df = pd.read_excel(allDomain_corrklmse_file, usecols=['Concept', corr_kl_mse])
        df_concept_type = df.copy()
        df_concept_super_category = df.copy()

        df_concept_type['concept_type'] = df_concept_type['Concept'].map(concept_type_dict).dropna()
        concept_type_group_stats = df_concept_type.groupby('concept_type').agg(['mean', 'std']).fillna(0)
        for concept_type in concept_type_group_stats.index:
            if LLM_name.endswith("-Instruct"):
                labelname = LLM_name.strip("-Instruct")
            else:
                labelname = LLM_name
            #
            # if labelname == "gpt-3.5-turbo":
            #     labelname = "GPT-3.5 Turbo"
            # if labelname == "gpt4omini":
            #     labelname = "GPT-4o mini"

            concept_type_data_list.append({
                "Model": labelname,
                "Concept Type": concept_type,
                "Mean " +corr_kl_mse: concept_type_group_stats.loc[concept_type, (corr_kl_mse, "mean")],
                "STD "+corr_kl_mse: concept_type_group_stats.loc[concept_type, (corr_kl_mse, "std")]
            })

        df_concept_super_category['concept_super_category'] = df_concept_super_category['Concept'].map(concept_super_category_dict).dropna()#因为用的不是所有的类型，所以有必要删除缺省行
        # print(df_concept_super_category)
        concept_super_category_group_stats = df_concept_super_category.groupby('concept_super_category').agg(['mean', 'std']).fillna(0)  # 按 concept_type 分组
        for concept_super_category in concept_super_category_group_stats.index:
            if LLM_name.endswith("-Instruct"):
                labelname = LLM_name.strip("-Instruct")
            else:
                labelname = LLM_name
            # if labelname == "gpt-3.5-turbo":
            #     labelname = "GPT-3.5 Turbo"
            # if labelname == "gpt4omini":
            #     labelname = "GPT-4o mini"
            concept_super_category_data_list.append({
                "Model": labelname,
                "Concept Super Category": concept_super_category,
                "Mean " + corr_kl_mse: concept_super_category_group_stats.loc[concept_super_category, (corr_kl_mse, "mean")],
                "STD " + corr_kl_mse: concept_super_category_group_stats.loc[concept_super_category, (corr_kl_mse, "std")]
            })

    # print(concept_type_data_list)
    # heatmap_plot(LLM_lst, concept_type_data_list, corr_kl_mse, "Concept Type", consistency_analysis_folder)
    # print(concept_super_category_data_list)
    heatmap_plot(LLM_lst, concept_super_category_data_list, corr_kl_mse, "Concept Super Category", consistency_analysis_folder)


def heatmap_plot(LLM_lst, concept_type_data_list, corr_kl_mse, conceptCat, consistency_analysis_folder):
    LLM_order_lst = []
    for llmname in LLM_lst:
        if llmname.endswith("-Instruct"):
            LLM_order_lst.append(llmname.strip("-Instruct"))
        else:
            if llmname == 'gpt-3.5-turbo':
                LLM_order_lst.append('GPT-3.5 Turbo')
            elif llmname == 'gpt4omini':
                LLM_order_lst.append('GPT-4o mini')


    concept_type_boxplot_df = pd.DataFrame(concept_type_data_list)
    concept_type_boxplot_df['Model'] = concept_type_boxplot_df['Model'].replace({'gpt-3.5-turbo': 'GPT-3.5 Turbo', 'gpt4omini': 'GPT-4o mini'})

    concept_type_boxplot_df['Model'] = pd.Categorical(concept_type_boxplot_df['Model'], categories=LLM_order_lst,
                                                      ordered=True)
    concept_type_boxplot_df = concept_type_boxplot_df.sort_values(by='Model')

    heatmap_data_mean = concept_type_boxplot_df.pivot(conceptCat, "Model", "Mean "+corr_kl_mse)
    heatmap_data_std = concept_type_boxplot_df.pivot(conceptCat, "Model", "STD "+corr_kl_mse)

    annot_data = heatmap_data_mean.copy()
    for row in annot_data.index:
        for col in annot_data.columns:
            mean_val = heatmap_data_mean.loc[row, col]
            std_val = heatmap_data_std.loc[row, col]
            annot_data.loc[row, col] = f"{mean_val:.2f}±{std_val:.2f}"


    my_colors = [(0.0, "#76a7c3"),
              (0.5, "#fff0de"),
              (1.0, "#c18586")]

    my_cmap = LinearSegmentedColormap.from_list("custom_gradient", my_colors)


    cmapname = "mycolors"
    plt.figure(figsize=(20, 12))
    sns.heatmap(
        heatmap_data_mean,
        annot=annot_data,
        fmt="",
        cmap=my_cmap,
        cbar_kws={'label': 'Mean '+corr_kl_mse},
        linewidths=0.3,
        annot_kws={"size": 12}
    )

    plt.title("Mean "+corr_kl_mse +" and STD Across Models and "+conceptCat)
    plt.xlabel("Model")
    plt.ylabel("Concept Type")
    annot_data.to_excel(consistency_analysis_folder +"BBSR_ConsistencyAnalysis_ALLmodels_"+conceptCat +"_" + corr_kl_mse + "_ANNOT.xlsx")
    plt.savefig(consistency_analysis_folder + "["+cmapname+"]BBSR_ConsistencyAnalysis_ALLmodels_"+conceptCat +"_" + corr_kl_mse + "_HEATMAP.png",
                bbox_inches='tight',
                dpi=300
                )


def load_concept_concaoa_dic(file_path):
    dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.split('\t')
            if value != "NA\n":
                dict[key.strip()] = float(value.strip())
    return dict

def plot_aoa_vs_corr_kl_mse(model_lst, output_folder, corr_kl_mse, model_colors_dic, aoa_window_size_lst):
    aoa_filepath = '../data/AoA30k.txt'
    concept_aoa_dic = load_concept_concaoa_dic(aoa_filepath)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for window_size in aoa_window_size_lst:
        plt.figure(figsize=(20, 12))
        sns.set(style="whitegrid")
        sns.set_context("talk")
        for i, model in enumerate(model_lst):
            concept_corr_kl_mse_file = '../results/BBSR_' + model + '/statistics/ALLDomain_corr_kl_mse.xlsx'
            df = pd.read_excel(concept_corr_kl_mse_file)

            df = df[df['Concept'].isin(concept_aoa_dic.keys())]
            df['AoA'] = df['Concept'].map(concept_aoa_dic)

            data_sorted = df.sort_values(by='AoA')

            data_sorted['corr_kl_mse_smoothed'] = uniform_filter1d(data_sorted[corr_kl_mse], size=window_size)
            if model.endswith("-Instruct"):
                labelname = model.strip("-Instruct")

            else:
                if model == "gpt-3.5-turbo":
                    labelname = "GPT-3.5 Turbo"
                if model == "gpt4omini":
                    labelname = "GPT-4o mini"
            plt.plot(data_sorted['AoA'], data_sorted['corr_kl_mse_smoothed'],
                     label=labelname, color=model_colors_dic[model], linewidth=4)
        plt.xlabel('Age of Acquisition (AoA)', fontsize=16, labelpad=10)
        corr_kl_mse_ylabel = {
            "corr": "Correlation",
            "mse": "Mean Squared Error",
            "kl": "Kullback-Leibler divergence"
        }.get(corr_kl_mse, "")

        plt.ylabel(corr_kl_mse_ylabel, fontsize=16, labelpad=10)
        plt.legend(
            loc='upper right',
            fontsize=16,
            title="Models",
            title_fontsize=18,
            ncol=2, #双列显示
            frameon=True,
            fancybox=True,
            shadow=True
        )

        plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)
        plt.grid(False, which='minor')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # plt.title(f"Smoothed {corr_kl_mse} vs. AoA\n(Window Size: {window_size})", fontsize=18, pad=15)
        plt.tight_layout()
        plt.savefig(output_folder+'AllConcepts_LinePlot_AoA_vs' + corr_kl_mse + '_win'+str(window_size) +'.png',dpi=200)
        plt.close()
def plot_conc_vs_corr_kl_mse(model_lst, output_folder, corr_kl_mse, model_colors_dic, conc_window_size_lst):
    conc_filepath = '../data/Concreteness40k.txt'
    concept_conc_dic = load_concept_concaoa_dic(conc_filepath)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for window_size in conc_window_size_lst:
        plt.figure(figsize=(20, 12))
        sns.set(style="whitegrid")
        sns.set_context("talk")
        for i, model in enumerate(model_lst):
            concept_corr_kl_mse_file = '../results/BBSR_' + model + '/statistics/ALLDomain_corr_kl_mse.xlsx'
            df = pd.read_excel(concept_corr_kl_mse_file)
            df = df[df['Concept'].isin(concept_conc_dic.keys())]
            df['Conc'] = df['Concept'].map(concept_conc_dic)

            data_sorted = df.sort_values(by='Conc')

            data_sorted['corr_kl_mse_smoothed'] = uniform_filter1d(data_sorted[corr_kl_mse], size=window_size)
            if model.endswith("-Instruct"):
                labelname = model.strip("-Instruct")
            else:
                if model == "gpt-3.5-turbo":
                    labelname = "GPT-3.5 Turbo"
                if model == "gpt4omini":
                    labelname = "GPT-4o mini"
            plt.plot(data_sorted['Conc'], data_sorted['corr_kl_mse_smoothed'],
                     label=labelname, color=model_colors_dic[model], linewidth=4)
        plt.xlabel('Concreteness', fontsize=16, labelpad=10)
        corr_kl_mse_ylabel = {
            "corr": "Correlation",
            "mse": "Mean Squared Error",
            "kl": "Kullback-Leibler divergence"
        }.get(corr_kl_mse, "")

        plt.ylabel(corr_kl_mse_ylabel, fontsize=16, labelpad=10)
        plt.legend(
            loc='upper left',
            fontsize=16,
            title="Models",
            title_fontsize=18,
            ncol=2,
            frameon=True,
            fancybox=True,
            shadow=True
        )

        plt.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.7)
        plt.grid(False, which='minor')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # plt.title(f"Smoothed {corr_kl_mse} vs. AoA\n(Window Size: {window_size})", fontsize=18, pad=15)
        plt.tight_layout()
        plt.savefig(output_folder+'AllConcepts_LinePlot_Conc_vs' + corr_kl_mse + '_win'+str(window_size) +'.png',dpi=200)
        plt.close()

def load_concept_concaoa_dic(file_path):
    dict = {}
    with open(file_path, 'r') as file:

        for line in file:
            key, value = line.split('\t')
            if value != "NA\n":
                dict[key.strip()] = float(value.strip())
    return dict

#################################################ConsistencyAnalysis#################################################

#################################################StabilityAnalysis#################################################
def sd_analysis(LLM_model_lst, sd_or_rsd_lst, sd_focus_domain_lst, model_colors_dic, stability_analysis_folder):
    if not os.path.exists(stability_analysis_folder):
        os.makedirs(stability_analysis_folder)
    dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()
    for sd_or_rsd in sd_or_rsd_lst:
        model_files = ['../results/BBSR_'+llm+'/statistics/'+llm+'_stability concept_'+sd_or_rsd+'.xlsx' for llm in LLM_model_lst]
        all_models_results = {domain: [] for domain in sd_focus_domain_lst}
        for file, model_name in zip(model_files, LLM_model_lst):
            data = pd.read_excel(file)
            for domain in sd_focus_domain_lst:
                dimensions = domain_dimlst_dic[domain]
                relevant_columns = data[dimensions].fillna(100)
                sd_or_rsd_mean = relevant_columns.values.mean()
                all_models_results[domain].append(sd_or_rsd_mean)
        x = np.arange(len(sd_focus_domain_lst))
        width =  0.8 / len(LLM_model_lst)
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model_name in enumerate(LLM_model_lst):
            if model_name.endswith("-Instruct"):
                labelname = model_name.strip("-Instruct")
            else:
                labelname = model_name
            color = model_colors_dic[model_name]
            ax.bar(x + i * width, [all_models_results[domain][i] for domain in sd_focus_domain_lst],
                   width, label=labelname, color=color)

        for i, model_name in enumerate(LLM_model_lst):
            model_results = [all_models_results[domain][i] for domain in sd_focus_domain_lst]
            model_x_positions = x + i * width

            if model_name.endswith("-Instruct"):
                labelname = model_name.strip("-Instruct")
            else:
                labelname = model_name
            color = model_colors_dic[model_name]
            ax.plot(model_x_positions, model_results, marker='o', linestyle='--', color=color)


        ax.set_xlabel("Domain")
        ax.set_ylabel(sd_or_rsd +" Mean")
        ax.set_title(sd_or_rsd +" Mean across Domains for Different Models")
        ax.set_xticks(x + width * (len(LLM_model_lst) - 1) / 2)
        ax.set_xticklabels(sa_focus_domain_lst)
        ax.legend(loc='best')

        plt.tight_layout()
        domain_info = "".join(sa_focus_domain_lst)
        plt.savefig(stability_analysis_folder+"ALLmodels_StabilityAnalysis_"+domain_info+"_"+sd_or_rsd+".png")



def get_domain_icc(domain_dimlst_dic, icc_focus_domain, test_file_lst):
    from pingouin import intraclass_corr

    domain_scores = {domain: [] for domain in icc_focus_domain}

    for file_path in test_file_lst:
        data = pd.read_excel(file_path, sheet_name="Sheet1")

        for domain, sub_dims in domain_dimlst_dic.items():
            if domain in icc_focus_domain:
                domain_score = data[sub_dims].mean(axis=1)
                domain_scores[domain].append(domain_score)


    icc_data = {domain: pd.DataFrame(np.array(scores).T, columns=[f"Test_{i}" for i in range(1, len(test_file_lst) + 1)])
                for domain, scores in domain_scores.items()}

    # ICC
    domain_icc_dic = {}
    for domain, df in icc_data.items():
        melted_df = df.reset_index().melt(id_vars="index", value_name="Score")
        icc = intraclass_corr(data=melted_df, targets="index", raters="variable", ratings="Score")
        icc_value = icc.loc[icc["Type"] == "ICC1", "ICC"].values[0]
        domain_icc_dic[domain] = icc_value
    return domain_icc_dic



def icc_analysis(LLM_model_lst, icc_focus_domain, stability_analysis_folder):
    if not os.path.exists(stability_analysis_folder):
        os.makedirs(stability_analysis_folder)
    dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()
    results_dic = {}
    for each_llm in LLM_model_lst:
        each_llm_result_folder = '../results/BBSR_'+ each_llm+'/origin/'
        all_files = os.listdir(each_llm_result_folder)
        testresults_file_list = [each_llm_result_folder + file for file in all_files if file.endswith('_scoreTTW.xlsx')]
        if each_llm == "Llama-3.1-405B-Instruct":
            testresults_file_list = [file for file in testresults_file_list if not file.endswith('5_scoreTTW.xlsx')]
        domain_icc_dic = get_domain_icc(domain_dimlst_dic, icc_focus_domain, testresults_file_list)
        results_dic[each_llm] = domain_icc_dic

    data = []
    for model in LLM_model_lst:
        row = [results_dic.get(model, {}).get(domain, "N/A") for domain in icc_focus_domain]
        data.append(row)
    df = pd.DataFrame(data, columns=icc_focus_domain, index=LLM_model_lst)
    df.to_excel(stability_analysis_folder+"ICC_results.xlsx", sheet_name="Results")




#################################################StabilityAnalysis#################################################



#################################################CloseConceptsAnalysis#################################################

def get_model_nearest_neighbors(df, concept, topk):
    distance_matrix = cosine_distances(df.values)
    if concept not in df.index:
        return f"Concept '{concept}' not found in the dataset."

    concept_idx = df.index.get_loc(concept)
    distances = distance_matrix[concept_idx]
    sorted_indices = distances.argsort()[1:topk + 1]

    neighbors = [df.index[i] for i in sorted_indices]
    return neighbors

def get_close_concepts(model_MEANresult_tuple_lst, topk, closeconcepts_folder):
    if not os.path.exists(closeconcepts_folder):
        os.makedirs(closeconcepts_folder)

    BBSR_file = "../data/[BBSR]WordSet1_Ratings.xlsx"
    dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()
    all65dims_lst = list(dim_domain_dic.keys())
    # load data
    human_data = (pd.read_excel(BBSR_file, usecols=['Word'] + all65dims_lst,
                                na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                  .rename(columns={'Word': 'Concept'})
                  .fillna(0)
                  .drop_duplicates()
                  .query("Concept != 'used'")
                  .set_index('Concept'))
    title_info = ["Concept", "Human"]+[tp[0] for tp in model_MEANresult_tuple_lst]
    results_nearest_neighbors_tp_lst = []
    for each_c in human_data.index.tolist():
        human_nearest_neighbors_lst = get_model_nearest_neighbors(human_data, each_c, topk)
        results_nearest_neighbors_tp_lst.append((each_c, "Human", ",".join(human_nearest_neighbors_lst) ))
        print((each_c, "Human", ",".join(human_nearest_neighbors_lst) ))
        for each_modelMEANresult_tuple in model_MEANresult_tuple_lst:
            LLM_name = each_modelMEANresult_tuple[0]
            LLM_MEAN_file = each_modelMEANresult_tuple[1]
            analysis_data_folder = '../results/BBSR_' + LLM_name + '/processed/'
            LLM_data = (pd.read_excel(analysis_data_folder + LLM_MEAN_file, usecols=['Concept'] + all65dims_lst,
                                      na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                        .fillna(0)
                        .drop_duplicates()
                        .query("Concept != 'used'")
                        .set_index('Concept'))
            LLM_nearest_neighbors_lst = get_model_nearest_neighbors(LLM_data, each_c, topk)
            results_nearest_neighbors_tp_lst.append((each_c, LLM_name, ",".join(LLM_nearest_neighbors_lst)))
            print((each_c, LLM_name, ",".join(LLM_nearest_neighbors_lst)))

    # Create a dictionary to store the data for the DataFrame
    data_dict = {title: [] for title in title_info}
    # Populate the dictionary
    for concept, model, neighbors in results_nearest_neighbors_tp_lst:
        if concept not in data_dict["Concept"]:
            data_dict["Concept"].append(concept)
            for col in title_info[1:]:
                data_dict[col].append("")
        idx = data_dict["Concept"].index(concept)
        data_dict[model][idx] = neighbors

    df = pd.DataFrame(data_dict)
    df.set_index("Concept", inplace=True)
    df.to_excel(closeconcepts_folder+str(topk)+"close_concepts.xlsx")

def get_precision_at_k_results(topk, closeconcepts_folder, precision_at_k_lst ):
    df = pd.read_excel(closeconcepts_folder+str(topk)+"close_concepts.xlsx")

    concepts = df['Concept']
    llm_columns = df.columns[1:]

    s_human_lst = defaultdict(list)
    s_llm_lst = defaultdict(lambda: defaultdict(list))

    for index, row in df.iterrows():
        concept = row['Concept']
        s_human_lst[concept] = row['Human'].split(',')
        for llm_column in llm_columns:
            s_llm_lst[concept][llm_column] = row[llm_column].split(',')

    precision_results = defaultdict(lambda: defaultdict(dict))

    for concept in concepts:
        for llm in llm_columns:
            for k in precision_at_k_lst:
                actual = s_human_lst[concept][:k]
                predicted = s_llm_lst[concept][llm][:k]
                # Calculate the overlap
                overlap = set(actual) & set(predicted)  # Intersection of both lists
                overlap_count = len(overlap)
                # Calculate Precision@k
                precision = overlap_count / k
                precision_results[concept][llm][f'Precision@{k}'] = precision

    for k in precision_at_k_lst:
        precision_data = {}
        # Iterate over concepts and LLMs
        for concept, llm_data in precision_results.items():
            row = {}
            for llm, k_values in llm_data.items():
                # Get the precision for the current k, or None if not available
                row[llm] = k_values.get('Precision@'+str(k), None)
            precision_data[concept] = row

        # Convert the precision data into a pandas DataFrame
        df = pd.DataFrame(precision_data).T  # Transpose to make concepts as index
        df = df.fillna(0)  # Optional: Replace None with 0 (or you can choose another value)

        df.to_excel(closeconcepts_folder+f'precision_results_at_{k}.xlsx')



    averages = defaultdict(dict)

    for llm in llm_columns:
        for k in precision_at_k_lst:
            avg_precision = sum(precision_results[concept][llm][f'Precision@{k}'] for concept in concepts) / len(
                concepts)
            averages[llm][f'meanPrecision@{k}'] = avg_precision

    data = []
    for llm, values in averages.items():
        row = {'LLM': llm}
        for k in precision_at_k_lst:
            row[f'meanPrecision@{k}'] = values.get(f'meanPrecision@{k}', None)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_excel(closeconcepts_folder + 'average_precision_results.xlsx', index=False)


#################################################ClosestConceptsAnalysis#################################################



if __name__ == "__main__":
    model_colors_dic =  {"Llama-3-8B-Instruct": "#87CEEB",
                        "Llama-3-70B-Instruct": "#4682B4",
                        "Llama-3-405B-Instruct": "#0F3557",
                        "Llama-3.1-8B-Instruct": "#D8BFD8",
                        "Llama-3.1-70B-Instruct": "#9370DB",
                        "Llama-3.1-405B-Instruct": "#5D3FD3",
                        "Qwen2-7B-Instruct": "#90EE90",
                        "Qwen2-72B-Instruct": "#3CB371",
                        "gpt-3.5-turbo": "#FFD580",
                        "gpt4omini": "#FF7F50"
                        }
    focus_colors = {
        "Vision": "#93b0cb",
        "Somatic": "#c3d4dd",
        "Audition": "#76a7c3",
        "Olfaction": "#A7BDD5",
        "Gustation": "#A3C9DC",

        "Spatial": "#d09291",
        "Temporal": "#c18586",

        "Causal": "#f3d7cc",
        "Social": "#e2eedf",
        "Emotion": "#fff0de",

        "Cognition": "#b15928",
        "Motor": "#e78ac3",
        "Drive": "#cab2d6",
        "Attention": "#fb9a99"
    }


    model_MEANresult_tuple_lst = [("Qwen2-7B-Instruct", "BBSR_Qwen2-7B-Instruct[MEAN]TTW.xlsx"),
                              ("Qwen2-72B-Instruct", "BBSR_Qwen2-72B-Instruct[MEAN]TTW.xlsx"),
                              ("Llama-3-8B-Instruct", "BBSR_Llama-3-8B-Instruct[MEAN]TTW.xlsx"),
                              ("Llama-3-70B-Instruct", "BBSR_Llama-3-70B-Instruct[MEAN]TTW.xlsx"),
                              ("Llama-3.1-8B-Instruct", "BBSR_Llama-3.1-8B-Instruct[MEAN]TTW.xlsx"),
                              ("Llama-3.1-70B-Instruct", "BBSR_Llama-3.1-70B-Instruct[MEAN]TTW.xlsx"),
                              ("Llama-3.1-405B-Instruct", "BBSR_Llama-3.1-405B-Instruct[MEAN]TTW.xlsx"),
                              ("gpt-3.5-turbo", "BBSR_gpt-3.5-turbo[MEAN]TTW.xlsx"),
                              ("gpt4omini", "BBSR_gpt-4o-mini[MEAN]TTW.xlsx")]

    toshow_folder = '../results/BBSR_FINAL_toshow/'
    if not os.path.exists(toshow_folder):
        os.makedirs(toshow_folder)
    multisensory_profile_folder = toshow_folder + "1.MultisensoryProfile/"
    consistency_analysis_folder = toshow_folder + "2.ConsistencyAnalysis/"
    stability_analysis_folder = toshow_folder + "3.StabilityAnalysis/"
    closeconcepts_folder = toshow_folder + "4.CloseConceptsAnalysis/"


    # Analysis1：multisensory profile
    human_color = "#6C757D"
    BBSR_file = "../data/[BBSR]WordSet1_Ratings.xlsx"
    rdm_concept_super_category_based_order = ["abstract entity", "abstract property", "abstract action",
                                            "artifact", "living object", "natural object",
                                            "event",
                                            "physical action", "physical property", "physical state",
                                            "mental state", "mental entity"]
    concepts_in_supercategory_order_lst = load_rdm_concept_order(BBSR_file, "Super Category", rdm_concept_super_category_based_order)

    rdm_concept_type_based_order = ["thing", "property", "action", "state"]
    concepts_in_type_order_lst = load_rdm_concept_order(BBSR_file, "Type", rdm_concept_type_based_order)

    supercategory_align_score_tuple_lst = []
    type_align_score_tuple_lst = []
    for each_modelMEANresult_tuple in model_MEANresult_tuple_lst:
        # 1)profile
        multisensory_profile_analysis(each_modelMEANresult_tuple[0], each_modelMEANresult_tuple[1], multisensory_profile_folder, human_color, model_colors_dic)

        # 2) RSA
        eachmodel_supercategory_align_score_tuple_lst = rsa_analysis(each_modelMEANresult_tuple[0],
                                                                     each_modelMEANresult_tuple[1],
                                                                     multisensory_profile_folder + "RSA_supercategory/",
                                                                     "Super Category",
                                                                     rdm_concept_super_category_based_order,
                                                                     concepts_in_supercategory_order_lst)
        supercategory_align_score_tuple_lst += eachmodel_supercategory_align_score_tuple_lst

        eachmodel_type_align_score_tuple_lst = rsa_analysis(each_modelMEANresult_tuple[0],
                                                                     each_modelMEANresult_tuple[1],
                                                                     multisensory_profile_folder + "RSA_type/",
                                                                     "Type",
                                                                      rdm_concept_type_based_order,
                                                                     concepts_in_type_order_lst)
        type_align_score_tuple_lst += eachmodel_type_align_score_tuple_lst

    output_align_score_results(supercategory_align_score_tuple_lst,
                               multisensory_profile_folder + "RSA_supercategory/supercategory_align_score_results.xlsx")
    output_align_score_results(type_align_score_tuple_lst,
                               multisensory_profile_folder + "RSA_type/type_align_score_results.xlsx")




    # Analysis2：Consistency Analysis（corr kl mse）
    corr_kl_mse_lst = ["mse", "corr"] #, "kl"
    ca_focus_domain_lst = ["Vision","Somatic", "Audition",
                           #"Olfaction", "Gustation",
                           "Spatial", "Temporal",
                           "Emotion", "Social", "Causal"]
                           # "Cognition", "Attention", "Drive","Motor"]#"Gustation", "Olfaction",

    LLM_lst = [t[0] for t in model_MEANresult_tuple_lst]

    filter_concept_type_lst = ["thing", "property", "action", "state"]
    filter_concept_super_category_lst = ["abstract entity", "abstract property", "abstract action",
                                         "artifact", "living object", "natural object"]

    aoa_window_size_lst = [100],
    LLM_lst_barplot = ["Qwen2-72B-Instruct", "Llama-3.1-405B-Instruct", "gpt4omini"]
    consistency_analysis(LLM_lst, corr_kl_mse_lst, ca_focus_domain_lst, focus_colors, consistency_analysis_folder,
                         filter_concept_type_lst, filter_concept_super_category_lst, model_colors_dic, aoa_window_size_lst)


    # Analysis3：Stability Analysis (sd rsd)

    LLM_model_lst = ["Qwen2-7B-Instruct",
                     "Qwen2-72B-Instruct",
                     "Llama-3-8B-Instruct",
                     "Llama-3-70B-Instruct",
                     "Llama-3.1-8B-Instruct",
                     "Llama-3.1-70B-Instruct",
                     "Llama-3.1-405B-Instruct",
                     "gpt-3.5-turbo",
                     "gpt4omini"]

    icc_focus_domain = ["Audition","Vision","Somatic",
                        "Olfaction", "Gustation",
                        "Spatial", "Temporal",
                        "Emotion", "Social", "Causal"]
    icc_analysis(LLM_model_lst, icc_focus_domain, stability_analysis_folder)



    # Analysis4： close concepts analysis
    topk = 20
    get_close_concepts(model_MEANresult_tuple_lst, topk, closeconcepts_folder)
    precision_at_k_lst = [1,3,5,10,15,20]
    get_precision_at_k_results(topk, closeconcepts_folder, precision_at_k_lst)









#!/usr/bin/env python
# -*- coding:utf-8 -*- #
# import openai
import time
import os
import csv
import pandas as pd
import openai


def load_BBSR_rating_df():
    path = '../data/[BBSR]WordSet1_Ratings.xlsx'
    df = pd.read_excel (path)
    concept_lst = df['Word'].tolist()
    return concept_lst, df

def load_BBSR_queries_df():
    path = '../data/[BBSR]Queries_v4.xlsx'
    N_df = pd.read_excel (path, sheet_name="Noun Queries")
    V_df = pd.read_excel (path, sheet_name="Verb Queries")
    A_df = pd.read_excel (path, sheet_name="Adj Queries")
    dims_lst = N_df['Name'].tolist()
    return dims_lst, N_df, V_df, A_df

def call_chat_completion(prompt):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[prompt]
            )
            return response.choices[0].message.content
        except Exception as e:

            time.sleep(10)




if __name__ == "__main__":
    BBSR_concept_lst, BBSR_rating_df = load_BBSR_rating_df()
    # print(BBSR_concept_lst)
    # print(len(BBSR_concept_lst))
    # print(BBSR_rating_df.loc[BBSR_rating_df['Word'] == 'actor', 'WC'].values[0])
    # print (BBSR_rating_df.loc[BBSR_rating_df['Word'] == 'actor', 'Vision'].values[0])
    # print (BBSR_rating_df.loc[BBSR_rating_df['Word'] == 'actor', 'Dark'].values[0])

    dims_lst, N_df, V_df, A_df = load_BBSR_queries_df ()
    # print(N_df.columns.tolist())
    # print(N_df.loc[N_df['Name'] == 'Dark', 'Query (To what degree do you think of this thing as…)'].values[0])
    # print(V_df.loc[V_df['Name'] == 'Dark', 'Query (To what degree do you think of this as…)'].values[0])
    # print(A_df.loc[A_df['Name'] == 'Dark', 'Query (To what degree do you think of this property as…)'].values[0])


    openai.api_base = "***"
    openai.api_key = "***"

    model = "gpt-4o-mini" # MODEL NAME
    user_id_lst = ["OpenAI-1", "OpenAI-2", "OpenAI-3", "OpenAI-4", "OpenAI-5"] # USER ID


    for each_userid in user_id_lst:
        prompt_output_file_path = "../results/"+"BBSR_"+model+"/" + "BBSR_" + model + "_" + each_userid + "_prompt.csv"
        score_output_file_path = "../results/" + "BBSR_" + model + "/" + "BBSR_" + model + "_" + each_userid + "_score.csv"
        f_score = open(score_output_file_path, "w")
        s_csv_writer = csv.writer (f_score)
        with open(prompt_output_file_path, 'w', encoding="utf8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Concept"] + dims_lst)
            s_csv_writer.writerow(["Concept"] + dims_lst)
            for each_concept in BBSR_concept_lst:
                print(each_concept)
                prompt_lst = []
                score_lst = []
                # Column C codes the word class of the item (1 = noun, 2 = verb, 3 = adjective).
                # Note that the word "used" was rated separately as a verb and as an adjective.
                if BBSR_rating_df.loc[BBSR_rating_df['Word'] == each_concept, 'WC'].values[0] == 1:
                    AVN_df = N_df
                    head = "To what degree do you think of this thing as "
                    content_dim = "Query (To what degree do you think of this thing as…)"
                elif BBSR_rating_df.loc[BBSR_rating_df['Word'] == each_concept, 'WC'].values[0] == 2:
                    AVN_df = V_df
                    head = "To what degree do you think of this as "
                    content_dim = "Query (To what degree do you think of this as…)"
                elif BBSR_rating_df.loc[BBSR_rating_df['Word'] == each_concept, 'WC'].values[0] == 3:
                    AVN_df = A_df
                    head = "To what degree do you think of this property as "
                    content_dim = "Query (To what degree do you think of this property as…)"
                else:
                    print("the word class of the item ERROR!")
                for each_dim in dims_lst:
                    relation = AVN_df.loc[AVN_df['Name'] == each_dim, 'Relation'].values[0]
                    content = AVN_df.loc[AVN_df['Name'] == each_dim, content_dim].values[0]

                    relation = " " if pd.isna (relation) else relation
                    content = " " if pd.isna (content) else content
                    dim_prompt = head + each_concept + " " + relation + " " + content + "? From 0 (not at all) to 6 (very much). " \
                                                                                   "Just provide the number, without any other output."

                    prompt = {"role": "user", "content": dim_prompt}
                    reponse_text = call_chat_completion(prompt).rstrip()

                    prompt_lst.append(dim_prompt)
                    score_lst.append(reponse_text)

                csv_writer.writerow([each_concept] + prompt_lst)
                s_csv_writer.writerow([each_concept] + score_lst)
                f.flush()
                f_score.flush()
                time.sleep(1)
        f_score.close()


                    













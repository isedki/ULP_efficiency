"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import os
import pandas as pd
from evaluation.utils.common import is_abstract, is_abstract2, is_abstrac2

def evaluate_template_level(dataset, groundtruth, parsedresult, output_dir):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    oracle_templates = list(pd.read_csv(groundtruth)['EventTemplate'].drop_duplicates().dropna())
    identified_templates = list(pd.read_csv(parsedresult)['EventTemplate'].drop_duplicates().dropna())
    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)

    comparison_results = []
    count =0
    for identified_template in identified_templates:
        identified_template_type = None

        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        #print(identified_template)
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        log_message_ids.drop_duplicates()
        log_message  = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template]
        #print(log_message)
        #print("size before",len(log_message['Content']))
        log_message = log_message.drop_duplicates(subset=['Content'])
        #print("size after",len(log_message['Content']))
        #if (len(log_message['Content']) <= 1 ) :
           # identified_template_type = "Unique"
           # count+=1

        #print(log_message_ids)
        num_messages = len(log_message_ids)
        #print("num message",num_messages)

        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)


        # incorrect template analysis
        if identified_template_type is None:

            # determine the template type
            template_types = set()
            for corr_oracle_template in corr_oracle_templates:
                ground_eventId = groundtruth_df.loc[groundtruth_df['EventTemplate'] == corr_oracle_template]
                sss= set(ground_eventId.EventId.tolist())
                event_id="NA"
                if len(sss) >0 :
                    event_id = list(sss)[0]
                        # Check SM (SaM
                same , partial = is_abstract(corr_oracle_template,identified_template)
                if same:
                    identified_template_type = 'SM'
                if partial :
                    identified_template_type = 'partial'
                #if is_abstract(identified_template, corr_oracle_template):
                    #template_types.add('OG')
                #elif is_abstract(corr_oracle_template, identified_template):
                    #template_types.add('UG')
                #else:
                    #template_types.add('MX')

            #if len(template_types) == 1:  # if the set is singleton
               # identified_template_type = template_types.pop()
            #else:
                #identified_template_type = 'MX'

        # save the results for the current identified template
        ids= log_message_ids.values.tolist()
        comparison_results.append([event_id,identified_template, identified_template_type, corr_oracle_templates, num_messages, ids])

    comparison_results_df = pd.DataFrame(comparison_results,
                                         columns=['EventId','identified_template', 'type', 'groundTruth', 'num_messages','log_ids'])
    comparison_results_df.sort_values(by=['EventId'])
    #print("+++++++",comparison_results_df[['EventId','num_messages']])
    same = comparison_results_df.loc[comparison_results_df['type'] == "SM"]
    partial = comparison_results_df.loc[comparison_results_df['type'] == "partial"]
    total_partial = partial['num_messages'].values.sum()
    #print(same)
    total_correct = same['num_messages'].values.sum()
    print ("total correct",total_correct)
    total_messages = len(parsedresult_df[['Content']])
    print("total messages", total_messages)
    PA= total_correct/ total_messages
    same = same.groupby(['EventId']).agg({'num_messages':'sum'})
    #print(same)
    oracle_templates_frame = pd.read_csv(groundtruth)
    #oracle_ = oracle_templates_frame.groupby(['EventId'])
    ground_data = oracle_templates_frame.groupby(['EventId']).count()
    ground_data.index.name = 'EventId'
    #ground_data.columns=['EventTemplate']
    ground_data.sort_values(by=['EventId'])
  
    #ground_data['EventId'] =  ground_data.index
    comparison_results_df['type'] = comparison_results_df['type'].fillna("No")
    #print(ground_data)
    #print(ground_data)
    #f3 = pd.concat([ground_data, same])
    df3 = pd.concat([ground_data, comparison_results_df])
    #print(df3)
    df3= pd.merge(ground_data,
                 comparison_results_df,
                 on='EventId',
                 how='left')
    #df3=comparison_results_df
    #df3['num_messages'] = df3['num_messages'].fillna(0)
    #df3 = df3[['EventTemplate','num_messages']]
    #df3['ommision'][df3['EventTemplate'] - df3['num_messages']>=0] = (df3['EventTemplate'] - df3['num_messages'])/df3['EventTemplate']
    df3 = df3.loc[df3['EventId'].astype(str).drop_duplicates().index]
    df3['num_messages'][df3['type'] =='No'] = 0
    df3['num_messages'][df3['type'] =='Unique'] = 0
    df3['num_messages'][df3['type'] !='SM'] = 0
    df3['num_messages'] = df3['num_messages'].fillna(0)
    #df3['partial'] = 
    df3['ommision'] = (df3['EventTemplate'] - df3['num_messages'])/df3['EventTemplate']
    df3['PAi'] = df3['num_messages']/df3['EventTemplate']
    df3['commission'] = (df3['num_messages'] - df3['EventTemplate'])/df3['num_messages']
    df3['commission'][df3['commission'] < 0] = 0
    df3['ommision'][df3['ommision'] < 0] = 0
    df3['ICSI'] = 1 - (df3['ommision']+ df3['commission'])
    #df3.drop_duplicates(subset=['EventTemplate'])
    
    df3 = df3[['EventId','identified_template','num_messages','groundTruth', 'EventTemplate','type','ommision','commission','ICSI','PAi']]
    pd.set_option('display.float_format', '{:.2f}%'.format)
    #print("nombre de template trouve",len(identified_templates))
    #print("count",count)
    if (len(oracle_templates)>=len(identified_templates)):
        std_glob = df3['ICSI'].sum()/(len(oracle_templates))
    else :
        std_glob = df3['ICSI'].sum()/(len(identified_templates))
    #print(df3)
    
    df3.to_csv(os.path.join(output_dir, dataset + '_results.csv'))
    
    #print("size of dataframe",df3['ommision'].size )
    #omi = df3['ommision'].sum()/df3['ommision'].size
    print("nb ommission:", df3['ommision'].sum()/df3['ommision'].size)
    if(len(oracle_templates)<len(identified_templates)):
        comi = (df3['commission'].sum()+len(identified_templates) - len(oracle_templates))/len(identified_templates)
        omi = (df3['ommision'].sum()+len(identified_templates) - len(oracle_templates))/len(identified_templates)
    else :
        comi = df3['commission'].sum()/ len(oracle_templates)
        omi = df3['ommision'].sum()/ len(oracle_templates)
    
    print("nb commissions:", comi)

    print("CSI score :", std_glob)
    print("PA :", PA)
    Average_PA=df3['PAi'].sum()/len(oracle_templates)
    print("average accuracy :", df3['PAi'].sum()/len(oracle_templates))
    
    #grouping_by_template = same.groupby(['corr_oracle_templates']).agg({'num_messages':'sum'})
    #print(grouping_by_template)
    comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA) = compute_template_level_accuracy(len(oracle_templates), df3,comparison_results_df )
    print("len(comparison_results_df[comparison_results_df.type == 'SM'])" ,len(df3[df3.type == 'SM']))
    print("len oracle template", len(oracle_templates))
    #print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}, OG: {:.4f}, UG: {:.4f}, MX: {:.4f}'.format(F1_measure, PTA, RTA, OG, UG, MX))
    print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}'.format(F1_measure, PTA, RTA))
    print('CSI', std_glob)
    return len(identified_templates), len(oracle_templates),PA, F1_measure, PTA, RTA, Average_PA,omi, comi, std_glob, len(same.index)/len(identified_templates), len(partial.index)/len(identified_templates)



def evaluate_template_level2(dataset, groundtruth, parsedresult, output_dir):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    oracle_templates = list(pd.read_csv(groundtruth)['EventTemplate'].drop_duplicates().dropna())
    identified_templates = list(pd.read_csv(parsedresult)['EventTemplate'].drop_duplicates().dropna())
    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)

    comparison_results = []
    count =0
    for identified_template in identified_templates:
        identified_template_type = None

        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        #print(identified_template)
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        log_message_ids.drop_duplicates()
        log_message  = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template]
        #print(log_message)
        #print("size before",len(log_message['Content']))
        log_message = log_message.drop_duplicates(subset=['Content'])
        #print("size after",len(log_message['Content']))
        #if (len(log_message['Content']) <= 1 ) :
           # identified_template_type = "Unique"
           # count+=1

        #print(log_message_ids)
        num_messages = len(log_message_ids)
        #print("num message",num_messages)

        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)


        # incorrect template analysis
        if identified_template_type is None:

            # determine the template type
            template_types = set()
            for corr_oracle_template in corr_oracle_templates:
                ground_eventId = groundtruth_df.loc[groundtruth_df['EventTemplate'] == corr_oracle_template]
                sss= set(ground_eventId.EventId.tolist())
                event_id="NA"
                if len(sss) >0 :
                    event_id = list(sss)[0]
                        # Check SM (SaM
                same , partial = is_abstract(corr_oracle_template,identified_template)
                if same:
                    identified_template_type = 'SM'
                elif partial :
                    identified_template_type = 'partial'
                #if is_abstract(identified_template, corr_oracle_template):
                    #template_types.add('OG')
                #elif is_abstract(corr_oracle_template, identified_template):
                    #template_types.add('UG')
                #else:
                    #template_types.add('MX')

            #if len(template_types) == 1:  # if the set is singleton
               # identified_template_type = template_types.pop()
            #else:
                #identified_template_type = 'MX'

        # save the results for the current identified template
        ids= log_message_ids.values.tolist()
        comparison_results.append([event_id,identified_template, identified_template_type, corr_oracle_templates, num_messages, ids])

    comparison_results_df = pd.DataFrame(comparison_results,
                                         columns=['EventId','identified_template', 'type', 'groundTruth', 'num_messages','log_ids'])
    comparison_results_df.sort_values(by=['EventId'])
    #print("+++++++",comparison_results_df[['EventId','num_messages']])
    same = comparison_results_df.loc[comparison_results_df['type'] == "SM"]
    partial = comparison_results_df.loc[comparison_results_df['type'] == "partial"]
    print(partial)
    total_partial = partial['num_messages'].values.sum()
    #print(same)
    total_correct = same['num_messages'].values.sum()
    print ("total correct",total_correct)
    total_messages = len(parsedresult_df[['Content']])
    print("total messages", total_messages)
    PA= total_correct/ total_messages
    same = same.groupby(['EventId']).agg({'num_messages':'sum'})
    #print(same)
    oracle_templates_frame = pd.read_csv(groundtruth)
    #oracle_ = oracle_templates_frame.groupby(['EventId'])
    ground_data = oracle_templates_frame.groupby(['EventId']).count()
    ground_data.index.name = 'EventId'
    #ground_data.columns=['EventTemplate']
    ground_data.sort_values(by=['EventId'])
  
    #ground_data['EventId'] =  ground_data.index
    comparison_results_df['type'] = comparison_results_df['type'].fillna("No")
    #print(ground_data)
    #print(ground_data)
    #f3 = pd.concat([ground_data, same])
    df3 = pd.concat([ground_data, comparison_results_df])
    #print(df3)
    df3= pd.merge(ground_data,
                 comparison_results_df,
                 on='EventId',
                 how='left')
    #df3=comparison_results_df
    #df3['num_messages'] = df3['num_messages'].fillna(0)
    #df3 = df3[['EventTemplate','num_messages']]
    #df3['ommision'][df3['EventTemplate'] - df3['num_messages']>=0] = (df3['EventTemplate'] - df3['num_messages'])/df3['EventTemplate']
    df3 = df3.loc[df3['EventId'].astype(str).drop_duplicates().index]
    df3['num_messages'][df3['type'] =='No'] = 0
    df3['num_messages'][df3['type'] =='Unique'] = 0
    df3['num_messages'][df3['type'] =='partial'] = 0
    #df3['num_messages'][df3['type'] =='SM'] = 1
    df3['num_messages'][df3['type'] ==''] = 0
    df3['num_messages'] = df3['num_messages'].fillna(0)
    #df3['partial'] = 
    df3['ommision'] = (df3['EventTemplate'] - df3['num_messages'])/df3['EventTemplate']
    df3['PAi'] = df3['num_messages']/df3['EventTemplate']
    df3['commission'] = (df3['num_messages'] - df3['EventTemplate'])/df3['num_messages']
    df3['commission'][df3['commission'] < 0] = 0
    df3['ommision'][df3['ommision'] < 0] = 0
    df3['ICSI'] = 1 - (df3['ommision']+ df3['commission'])
    #df3.drop_duplicates(subset=['EventTemplate'])
    
    df3 = df3[['EventId','identified_template','num_messages','groundTruth', 'EventTemplate','type','ommision','commission','ICSI','PAi']]
    pd.set_option('display.float_format', '{:.2f}%'.format)
    #print("nombre de template trouve",len(identified_templates))
    #print("count",count)
    if (len(oracle_templates)>=len(identified_templates)):
        std_glob = df3['ICSI'].sum()/(len(oracle_templates))
    else :
        std_glob = df3['ICSI'].sum()/(len(identified_templates))
    #print(df3)
    
    df3.to_csv(os.path.join(output_dir, dataset + '_results.csv'))
    
    #print("size of dataframe",df3['ommision'].size )
    #omi = df3['ommision'].sum()/df3['ommision'].size
    print("nb ommission:", df3['ommision'].sum()/df3['ommision'].size)
    if(len(oracle_templates)<len(identified_templates)):
        comi = (df3['commission'].sum()+len(identified_templates) - len(oracle_templates))/len(identified_templates)
        omi = (df3['ommision'].sum()+len(identified_templates) - len(oracle_templates))/len(identified_templates)
    else :
        comi = df3['commission'].sum()/ len(oracle_templates)
        omi = df3['ommision'].sum()/ len(oracle_templates)
    
    print("nb commissions:", comi)

    print("CSI score :", std_glob)
    print("PA :", PA)
    Average_PA=df3['PAi'].sum()/len(oracle_templates)
    print("average accuracy :", df3['PAi'].sum()/len(oracle_templates))
    
    #grouping_by_template = same.groupby(['corr_oracle_templates']).agg({'num_messages':'sum'})
    #print(grouping_by_template)
    comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA) = compute_template_level_accuracy(len(oracle_templates), df3,comparison_results_df )
    print("len(comparison_results_df[comparison_results_df.type == 'SM'])" ,len(df3[df3.type == 'SM']))
    print("len oracle template", len(oracle_templates))
    #print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}, OG: {:.4f}, UG: {:.4f}, MX: {:.4f}'.format(F1_measure, PTA, RTA, OG, UG, MX))
    print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}'.format(F1_measure, PTA, RTA))
    print('CSI', std_glob)
    return len(identified_templates), len(oracle_templates),PA, F1_measure, PTA, RTA, Average_PA,omi, comi, std_glob, len(same.index)/len(identified_templates), len(partial.index)/len(identified_templates)





def find_corr_oracle_templates(log_message_ids, groundtruth_df):
    """
    Identify the corresponding oracle templates for the tool-generated(identified) template

    :param log_message_ids: Log_message ids that corresponds to the tool-generated template
    :param groundtruth_df: Oracle structured file
    :return: Identified oracle templates that corresponds to the tool-generated(identified) template
    """

    corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
    corresponding_oracle_templates = list(corresponding_oracle_templates.EventTemplate.unique())
    return corresponding_oracle_templates


def compute_template_level_accuracy(num_oracle_template,df3, comparison_results_df):
    """Calculate the template-level accuracy values.

    :param num_oracle_template: total number of oracle templates
    :param comparison_results_df: template analysis results (dataFrame)
    :return: f1, precision, recall
    """
    count_total = float(len(comparison_results_df))
    print("O))))))))))))",len(comparison_results_df[comparison_results_df.type == 'SM']))
    precision = len(df3[df3.type == 'SM']) / count_total  # PTA
    recall = len(df3[df3.type == 'SM']) / float(num_oracle_template)  # RTA
    f1_measure = 0.0
    if precision != 0 or recall != 0:
        f1_measure = 2 * (precision * recall) / (precision + recall)
    return f1_measure, precision, recall

def evaluate_template_level2(dataset, groundtruth, parsedresult, output_dir):
    """
    Conduct the template-level analysis using 4-type classifications

    :param dataset:
    :param groundtruth:
    :param parsedresult:
    :param output_dir:
    :return: SM, OG, UG, MX
    """

    oracle_templates = list(pd.read_csv(groundtruth)['EventTemplate'].drop_duplicates().dropna())
    identified_templates = list(pd.read_csv(parsedresult)['EventTemplate'].drop_duplicates().dropna())
    parsedresult_df = pd.read_csv(parsedresult)
    groundtruth_df = pd.read_csv(groundtruth)

    comparison_results = []
    for identified_template in identified_templates:
        identified_template_type = None

        # Get the log_message_ids corresponding to the identified template from the tool-generated structured file
        log_message_ids = parsedresult_df.loc[parsedresult_df['EventTemplate'] == identified_template, 'LineId']
        log_message_ids = pd.DataFrame(log_message_ids)
        num_messages = len(log_message_ids)

        # identify corr_oracle_templates using log_message_ids and oracle_structured_file
        corr_oracle_templates = find_corr_oracle_templates(log_message_ids, groundtruth_df)

        # Check SM (SaMe)
        if set(corr_oracle_templates) == {identified_template}:
            identified_template_type = 'SM'

        # incorrect template analysis
        if identified_template_type is None:

            # determine the template type
            template_types = set()
            for corr_oracle_template in corr_oracle_templates:
                if is_abstrac2(identified_template, corr_oracle_template):
                    template_types.add('OG')
                elif is_abstrac2(corr_oracle_template, identified_template):
                    template_types.add('UG')
                else:
                    template_types.add('MX')

            if len(template_types) == 1:  # if the set is singleton
                identified_template_type = template_types.pop()
            else:
                identified_template_type = 'MX'

        # save the results for the current identified template
        comparison_results.append([identified_template, identified_template_type, corr_oracle_templates, num_messages])

    comparison_results_df = pd.DataFrame(comparison_results,
                                         columns=['identified_template', 'type', 'corr_oracle_templates', 'num_messages'])
    comparison_results_df.to_csv(os.path.join(output_dir, dataset + '_template_analysis_results.csv'), index=False)
    (F1_measure, PTA, RTA, OG, UG, MX) = compute_template_level_accuracy2(len(oracle_templates), comparison_results_df)
    print('F1: {:.4f}, PTA: {:.4f}, RTA: {:.4f}, OG: {:.4f}, UG: {:.4f}, MX: {:.4f}'.format(F1_measure, PTA, RTA, OG, UG, MX))
    return len(identified_templates), len(oracle_templates), F1_measure, PTA, RTA, OG, UG, MX


def find_corr_oracle_templates2(log_message_ids, groundtruth_df):
    """
    Identify the corresponding oracle templates for the tool-generated(identified) template

    :param log_message_ids: Log_message ids that corresponds to the tool-generated template
    :param groundtruth_df: Oracle structured file
    :return: Identified oracle templates that corresponds to the tool-generated(identified) template
    """

    corresponding_oracle_templates = groundtruth_df.merge(log_message_ids, on='LineId')
    corresponding_oracle_templates = list(corresponding_oracle_templates.EventTemplate.unique())
    return corresponding_oracle_templates


def compute_template_level_accuracy2(num_oracle_template, comparison_results_df):
    """Calculate the template-level accuracy values.

    :param num_oracle_template: total number of oracle templates
    :param comparison_results_df: template analysis results (dataFrame)
    :return: f1, precision, recall, over_generalized, under_generalized, mixed
    """
    count_total = float(len(comparison_results_df))
    precision = len(comparison_results_df[comparison_results_df.type == 'SM']) / count_total  # PTA
    recall = len(comparison_results_df[comparison_results_df.type == 'SM']) / float(num_oracle_template)  # RTA
    over_generalized = len(comparison_results_df[comparison_results_df.type == 'OG']) / count_total
    under_generalized = len(comparison_results_df[comparison_results_df.type == 'UG']) / count_total
    mixed = len(comparison_results_df[comparison_results_df.type == 'MX']) / count_total
    f1_measure = 0.0
    if precision != 0 or recall != 0:
        f1_measure = 2 * (precision * recall) / (precision + recall)
    return f1_measure, precision, recall, over_generalized, under_generalized, mixed


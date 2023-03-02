"""
Description : This file implements the LogMine algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import sys
import re
import os

import copy
import hashlib
import pandas as pd
from datetime import datetime
from collections import defaultdict

class partition():
    def __init__(self, idx, log="", lev=-1):
        self.logs_idx = [idx]
        self.patterns = [log]
        self.level = lev

class LogParser():
    def __init__(self, indir, outdir, log_format, max_dist=0.001, levels=2, k=1, k1=1, k2=1, alpha=100, rex=[]):
        self.logformat = log_format
        self.path = indir
        self.savePath = outdir
        self.rex = rex
        self.levels = levels
        self.max_dist = max_dist
        self.k = k
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.df_log = None
        self.logname = None
        self.level_clusters = {}


    def parse(self, logname):
        print('Parsing file: ' + os.path.join(self.path, logname))
        self.logname = logname
        starttime = datetime.now()
        self.load_data()
        for lev in range(self.levels):
            if lev == 0:
                # Clustering
                self.level_clusters[0] = self.get_clusters(self.df_log['Content_'], lev)
            else:
                # Clustering
                patterns = [c.patterns[0] for c in self.level_clusters[lev-1]]
                self.max_dist *= self.alpha
                clusters = self.get_clusters(patterns, lev, self.level_clusters[lev-1])

                # Generate patterns
                for cluster in clusters:
                    cluster.patterns = [self.sequential_merge(cluster.patterns)]
                self.level_clusters[lev] = clusters
        self.dump()
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

    def dump(self):
        if not os.path.isdir(self.savePath):
            os.makedirs(self.savePath)

        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        templates_occ = defaultdict(int)
        for cluster in self.level_clusters[self.levels-1]:
            EventTemplate = cluster.patterns[0]
            EventId = hashlib.md5(' '.join(EventTemplate).encode('utf-8')).hexdigest()[0:8]
            Occurences = len(cluster.logs_idx)
            templates_occ[EventTemplate] += Occurences

            for idx in cluster.logs_idx:
                ids[idx] = EventId
                templates[idx]= EventTemplate
        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['Occurrences'] = self.df_log['EventTemplate'].map(occ_dict)
        df_event['EventId'] = self.df_log['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[0:8])

        self.df_log.drop("Content_", inplace=True, axis=1)
        self.df_log.to_csv(os.path.join(self.savePath, self.logname + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False, columns=["EventId","EventTemplate","Occurrences"])

    def get_clusters(self, logs, lev, old_clusters=None):
        clusters = []
        old_clusters = copy.deepcopy(old_clusters)
        for logidx, log in enumerate(logs):
            match = False
            for cluster in clusters:
                dis = self.msgDist(log, cluster.patterns[0]) if lev == 0 else self.patternDist(log, cluster.patterns[0])
                if dis and dis < self.max_dist:
                    if lev == 0:
                        cluster.logs_idx.append(logidx)
                    else:
                        cluster.logs_idx.extend(old_clusters[logidx].logs_idx)
                        cluster.patterns.append(old_clusters[logidx].patterns[0])
                    match = True

            if not match: 
                if lev == 0:
                    clusters.append(partition(logidx, log, lev)) # generate new cluster
                else:
                    old_clusters[logidx].level = lev
                    clusters.append(old_clusters[logidx]) # keep old cluster

        return clusters

    def sequential_merge(self, logs):
        log_merged = logs[0]
        for log in logs[1:]:
            log_merged = self.pair_merge(log_merged, log)
        return log_merged

    def pair_merge(self, loga, logb):
        loga, logb = water(loga.split(), logb.split())
        logn = []
        for idx, value in enumerate(loga):
            logn.append('<*>' if value != logb[idx] else value)
        return " ".join(logn)

    def print_cluster(self, cluster):
        print ("------start------")


    def msgDist(self, seqP, seqQ):
        dis = 1
        seqP = seqP.split()
        seqQ = seqQ.split()
        maxlen = max(len(seqP), len(seqQ))
        minlen = min(len(seqP), len(seqQ))
        for i in range(minlen):
            dis -= (self.k if seqP[i]==seqQ[i] else 0 * 1.0) / maxlen
        return dis

    def patternDist(self, seqP, seqQ):
        dis = 1
        seqP = seqP.split()
        seqQ = seqQ.split()
        maxlen = max(len(seqP), len(seqQ))
        minlen = min(len(seqP), len(seqQ))
        for i in range(minlen):
            if seqP[i] == seqQ[i]:
                if seqP[i] == "<*>":
                    dis -= self.k2 * 1.0 / maxlen
                else:
                    dis -= self.k1 * 1.0 / maxlen
        return dis

    def load_data(self):
        def preprocess(line):
            for currentRex in self.rex:
                line = re.sub(currentRex, '', line)
            return line

        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logname), regex, headers, self.logformat)
        self.df_log['Content_'] = self.df_log['Content'].map(preprocess)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        ''' Function to transform log file to dataframe '''
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        ''' 
        Function to generate regular expression to split log messages
        '''
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
    # This software is a free software. Thus, it is licensed under GNU General Public License.
# Python implementation to Smith-Waterman Algorithm for Homework 1 of Bioinformatics class.
# Forrest Bao, Sept. 26 <http://fsbao.net> <forrest.bao aT gmail.com>

# zeros() was origianlly from NumPy.
# This version is implemented by alevchuk 2011-04-10
def zeros(shape):
    retval = []
    for x in range(shape[0]):
        retval.append([])
        for y in range(shape[1]):
            retval[-1].append(0)
    return retval

match_award      = 10
mismatch_penalty = 1
gap_penalty      = 0 # both for opening and extanding

def match_score(alpha, beta):
    if alpha == beta:
        return match_award
    elif alpha == '-' or beta == '-':
        return gap_penalty
    else:
        return mismatch_penalty

def finalize(align1, align2):
    align1.reverse()    #reverse sequence 1
    align2.reverse()    #reverse sequence 2

    i,j = 0,0
    
    #calcuate identity, score and aligned sequeces
    symbol = ''
    found = 0
    score = 0
    identity = 0
    for i in range(0,len(align1)):
        # if two AAs are the same, then output the letter
        if align1[i] == align2[i]:                
            symbol = symbol + align1[i]
            identity = identity + 1
            score += match_score(align1[i], align2[i])
    
        # if they are not identical and none of them is gap
        elif align1[i] != align2[i] and align1[i] != '-' and align2[i] != '-': 
            score += match_score(align1[i], align2[i])
            symbol += ' '
            found = 0
    
        #if one of them is a gap, output a space
        elif align1[i] == '-' or align2[i] == '-':          
            symbol += ' '
            score += gap_penalty
    
    identity = float(identity) / len(align1) * 100
    
    return align1, align2

def water(seq1, seq2):
    m, n = len(seq1), len(seq2)  # length of two sequences
    
    # Generate DP table and traceback path pointer matrix
    score = zeros((m+1, n+1))      # the DP table
    pointer = zeros((m+1, n+1))    # to store the traceback path
    
    max_score = 0        # initial maximum score in DP table
    # Calculate DP table and mark pointers
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            score_diagonal = score[i-1][j-1] + match_score(seq1[i-1], seq2[j-1])
            score_up = score[i][j-1] + gap_penalty
            score_left = score[i-1][j] + gap_penalty
            score[i][j] = max(0,score_left, score_up, score_diagonal)
            if score[i][j] == 0:
                pointer[i][j] = 0 # 0 means end of the path
            if score[i][j] == score_left:
                pointer[i][j] = 1 # 1 means trace up
            if score[i][j] == score_up:
                pointer[i][j] = 2 # 2 means trace left
            if score[i][j] == score_diagonal:
                pointer[i][j] = 3 # 3 means trace diagonal
            if score[i][j] >= max_score:
                max_i = i
                max_j = j
                max_score = score[i][j];
    
    align1 = []
    align2 = []    # initial sequences
    
    i,j = max_i,max_j    # indices of path starting point
    
    #traceback, follow pointers
    while pointer[i][j] != 0:
        if pointer[i][j] == 3:
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif pointer[i][j] == 2:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1
        elif pointer[i][j] == 1:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1

    return finalize(align1, align2)

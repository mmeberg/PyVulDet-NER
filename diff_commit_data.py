# This script is part of a larger research project shown at https://github.com/mmeberg/PyVulDet-NER
# The code below referenced the work done in https://github.com/LauraWartschinski/VulnerabilityDetection to obtain GitHub samples. 

import myutils
import time
import mytokenizer
import sys
import json
import subprocess
from datetime import datetime
import requests 
import pickle
from pydriller import Repository
from collections import Counter
import signal
import os
 

def getChanges(rest):
    changes = []
    while ("diff --git" in rest):
        filename = ""
        start = rest.find("diff --git")+1
        secondpart = rest.find("index")
        #get the title line which contains the file name
        titleline = rest[start:secondpart]
        if ".py" not in rest[start:secondpart] and "python" not in rest[start:secondpart].lower() and '.ipynb' not in rest[start:secondpart]:
            # No python file changed in this part of the commit
            rest = rest[secondpart+1:]
            continue
        
        if "diff --git" in rest[start:]:
            end = rest[start:].find("diff --git");
            filechange = rest[start:end]
            rest = rest[end:]
        else:
            end = len(rest)
            filechange = rest[start:end]
            rest = ""
        filechangerest = filechange

        while ("@@" in filechangerest):
            #singular changes are marked by @@ 
            change = ""
            start = filechangerest.find("@@")+2
            start2 = filechangerest[start:start+50].find("@@")+2
            start = start+start2
            filechangerest=filechangerest[start:]

            if ("class" in filechangerest or "def" in filechangerest) and "\n" in filechangerest:
                filechangerest = filechangerest[filechangerest.find("\n"):]

            if "@@" in filechangerest:
                end = filechangerest.find("@@")
                change = filechangerest[:end]
                filechangerest = filechangerest[end+2:]
            else:
                end = len(filechangerest)
                change = filechangerest[:end]
                filechangerest = ""

            if len(change) > 0:
                changes.append([titleline,change])

    return changes


def getFilename(titleline):
    #extracts the file name from the title line of a diff file
    s = titleline.find(" a/")+2
    e = titleline.find(" b/")
    name = titleline[s:e]

    if titleline.count(name) == 2:
        return name
    elif ".py" in name and (" a"+name+" " in titleline):
        return name

    
def getBadpart(change):
    badexamples = []
    goodexamples = []
    lines = change.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if line[0] == "-":
                badexamples.append(line[1:])
            if line[0] == "+":
                goodexamples.append(line[1:])
    if len(badexamples) == 0:
        return None
    else:
        return [badexamples,goodexamples]

    
def getDiffOrig(change):
    difforig = ''
    lines = change.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if line[0] == "-":
                difforig = difforig + '\n' + line[1:]
            elif line[0] == "+":
                continue
            else:
                difforig = difforig + '\n' + line
    return difforig
    
    
def makechangeobj(changething):
    change = changething[1]
    titleline = changething[0]
    thischange = {}

    if getBadpart(change) != None:      
        badparts = getBadpart(change)[0]
        goodparts = getBadpart(change)[1]
        linesadded = change.count("\n+")
        linesremoved = change.count("\n-")
        thischange["diff"] = change
        thischange["add"] = linesadded
        thischange["remove"] = linesremoved
        thischange["filename"] = getFilename(titleline)
        thischange["diffOrig"] = getDiffOrig(change)
        thischange["badparts"] = badparts
        thischange["goodparts"] = []
        if goodparts != None:
            thischange["goodparts"] = goodparts
        return thischange
    else:
        return None



chunked_file_list = list(os.listdir('ChunkedData\\'))
file_list = ['ChunkedData\\'+ str(f) for f in chunked_file_list]
file_list.append('commits_with_diffs.json')

keywords_dict = {'oob':['Out-of-bounds write','out of bounds write', 'out of bounds',  'memory corruption', 'CWE-787',
                        'intended buffer', 'out-of-bounds', 'allows heap corruption', 'allows memory corruption', 
                        'browser allows heap corruption', 'user can obtain read/write access to read-only pages', 
                        'out-of-bounds write (CWE-787)', 'Out-of-bounds write in kernel-mode driver', 
                        'incorrect bounds check', 'Memory corruption in web browser scripting engine', 
                        'leading to out-of-bounds write', 'allowing out-of-bounds write', 
                        'leading to memory corruption', 'leads to buffer underflow',  
                        'stack-based buffer overflow', 'Heap-based buffer overflow' 
                       ],
                 
                 'xss':['Cross-site Scripting', 'cross site', 'cross-site','CWE-79', 'Improper Neutralization of Input During Web Page Generation',
                        'XSS', 'HTML injection', 'contains untrusted data', 'executable by a web browser', 'untrusted data', 
                        'web browser executes the malicious script', 'Reflected XSS', 'Stored XSS', 'DOM-Based XSS',
                        'prevent XSS', 'prevent cross site', 'fix XSS', 'fix cross site', 'correct XSS', 'correct cross site',
                        'allow reflected XSS', 'allowing reflected XSS', 'allowed reflected XSS',
                        'reflected Cross-Site Scripting attacks','XSS (CWE-79)', 'allows XSS', 'insert malicious HTML sequences', 'XSS flaw',
                        'did not sufficiently neutralize', 'allowing for reflected Cross-Site Scripting attacks', 
                        'Universal XSS', 'Admin GUI allows XSS through cookie', 'allows XSS through crafted HTTP header',
                        'allows XSS through crafted HTTP Referer header', 'protection mechanism failure allows XSS', 
                        'allowing XSS (CWE-79) using other tags', 'enabling XSS (CWE-79)', 'Reflected XSS using the PATH_INFO in a URL',
                        'Stored XSS in a security product', 'Stored XSS using a wiki page', 'Stored XSS in a guestbook application',                   
                       ],
                 
                 'sql':['Improper Neutralization of Special Elements used in an SQL Command', 'CWE-89', 
                        'SQL Injection','SQL Command', 'improper SQL syntax', 'prevent sql injection'
                       ], 
                 
                 'iiv':['CWE-20', 'Improper Input Validation', 'does not validate input',
                        'incorrectly validates the input', 'does not validate input properly',
                        'input validation', 'bypass a validation step', 
                        'insufficient input validation', 'improved input validation'
                       ],
                 
                 'rce':['Improper Control of Generation of Code', 'Code Injection', 'RCE',  'CWE-94',
                        'remote code execution', 'modify the syntax', 'modify the behavior',
                        'alter the intended control flow of the software', 'arbitrary code execution', 
                        'injection weakness', 'string vulnerabilities',
                        'prevent rce','prevent remote code execution', 'fix rce',
                        'fix remote code execution', 'correct rce','correct remote code execution'
                       ],
                 
                 'pat':['CWE-22', 'Improper Limitation of a Pathname to a Restricted Directory','Path Traversal',
                        'directory traversal', 'outside the restricted directory', 
                        'escape outside of the restricted location', 'relative path traversal', 
                        'absolute path traversal', 'accessing unexpected files', 
                        'injection of a null byte', 'truncate a generated filename', 'null injection',
                        'prevent directory traversal', 'fix directory traversal', 'correct directory traversal',
                        'absolute pathname', 'drive letter'],
                } 
mode_list = ["sql", "oob", 'iiv',"rce", 'pat', "xss"] 

now1 = datetime.now() # current date and time
nowformat1 = now1.strftime("%H:%M")
print("start ", nowformat1)

progress = 0
changelist = []
cve_types = set()
keyword_test = list()

for file in file_list:
    print(file)
    
    version = file.split('\\')[-1].split('_')[-1].split('.')[0]
    
    with open(file, 'r') as infile:
        data_orig = json.load(infile)
    print('length of data: ', len(data_orig))
    
    for mode in mode_list:
        print(mode)
        filename = mode + '_' + version +'.json'
        if filename in list(os.listdir('Data\\')):
            print(filename, ' already in folder')
            continue
        keywords_to_check = [x.lower() for x in keywords_dict[mode]]
        issues = 0
        data_new_list = []

        for r in data_orig:
            for c in data_orig[r]:
                
                keyword_test.append(data_orig[r][c]["keyword"])   

                if data_orig[r][c]["keyword"].lower() not in keywords_to_check:
                    #doesn't contain the keyword we need for the specific mode
                    continue

                #check for duplicates 
                if c not in changelist:
                    changelist.append(c)
                else: 
                    #commit already found
                    continue

                #get all changes in the diff file
                changes = getChanges(data_orig[r][c]["diff"])
                for change in changes:
                    #make them into usable objects
                    thischange = makechangeobj(change)
                    if thischange != None:
                        data_new= {}
                        data_new["keyword"] = data_orig[r][c]["keyword"]
                        data_new["cwetype"] = mode
                        data_new['repo'] = r
                        data_new['commit'] = c
                        data_new['goodparts'] = thischange['goodparts']
                        data_new['badparts'] = thischange['badparts']
                        data_new['orig_diff'] = thischange['diff']
                        data_new['orig_txt'] = thischange["diffOrig"] 
                        data_new_list.append(data_new)

        print("done.")
        print('len of data: ', len(data_new_list))
        
        #save dataset
        with open('Data\\'+ mode + '_' + version +'.json', 'w') as outfile:
            json.dump(data_new_list, outfile)

# This script is part of a larger research project shown at https://github.com/mmeberg/PyVulDet-NER
# The code below referenced the work done in https://github.com/LauraWartschinski/VulnerabilityDetection to obtain GitHub samples. 

import requests
import time
import sys
import json
import datetime


def check_if_have(repository, sha):
    if repository in prev_results: 
        if sha in prev_results[repository]:
            if 'diff' in prev_results[repository][sha]:
                return True
            else:
                return False
        else:
            return False
    else:
        return False

def getdiffs(repository):
    repo = repository
    for c in repositories[repo]:
        alreadyhaveit = check_if_have(repo, c)
        
        if alreadyhaveit == True:
            if repo not in data:
                data[repo] = {}
            if c not in data[repo]:
                data[repo][c] = {}
            data[repo][c]["sha"] = prev_results[repo][c]["sha"]
            data[repo][c]["keyword"] = prev_results[repo][c]["keyword"]
            data[repo][c]["diff"] = prev_results[repo][c]["diff"]

        else:
            # get commit diff
            target = repo+'/commit/' + c + '.diff'
            myheaders = {'Authorization': 'token ' + access}
            try:
                response = requests.get(target,headers = myheaders)
            except:
                print('had an issue with response')
                continue
            content = response.content
            try:
                diffcontent = content.decode('utf-8')
            except:
                diffcontent = content.decode("latin-1")

            #check if the file contains any python
            if ".py" in diffcontent:
                if repo not in data:
                    data[repo] = {} 
                if c not in data[repo]:
                    data[repo][c] = {}
                
                data[repo][c]['sha'] = repositories[repo][c]['sha']
                data[repo][c]['keyword'] = repositories[repo][c]['keyword']
                data[repo][c]["diff"] = diffcontent
        
    return data



start = time.time()

if not os.path.isfile('access'):
    print("Need a Github access token in this directory.")
    sys.exit()
with open('access', 'r') as accestoken:
    access = accestoken.readline().replace("\n","")

#Commits from "commits_from_GitHub.py"
repositories = {}
with open('all_commits.json', 'r') as infile:
    repositories = json.load(infile)
print('# of repositories: ', len(repositories))

# Results from this running before
prev_results = {}
if os.path.isfile('commits_with_diffs.json'):
    with open('commits_with_diffs.json', 'r') as infile:
        prev_results = json.load(infile)

data = {}

for idx, r in enumerate(repositories):
    data = getdiffs(r)
    if idx%10 == 0:
        print(idx)
    if len(data)%100 == 0:
        print(idx, len(data))
        print("saving part of data")
        with open("ChunkedData\\"+"part_of_diffs_data_"+str(idx)+".json", 'w') as outfile:
            json.dump(data, outfile)
        data = {}
    elif idx == len(repositories)-1:
        print(idx, len(data))
        print("saving part of data")
        with open("ChunkedData\\"+"part_of_diffs_data_"+str(idx)+".json", 'w') as outfile:
            json.dump(data, outfile)
        data = {}

print('len(data): ', len(data))
with open('commits_with_diffs.json', 'w') as outfile:
    json.dump(data, outfile)

    
end = time.time()
elapsed = (end-start)/60
print('time elapsed: ', elapsed)

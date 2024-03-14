# This script is part of a larger research project shown at https://github.com/mmeberg/PyVulDet-NER

import os
import requests
import time
import sys
import json
from requests_oauthlib import OAuth1Session
from requests_oauthlib import OAuth1
import datetime

def searchforkeyword(key, commits, access):
    params = (('q', key+'+language:python+committer-date:>2012-01-01'),
              ('per_page',100), 
              ('sort', 'committer-date'))
    myheaders = {'Accept': 'application/vnd.github.cloak-preview', 'Authorization': 'token ' + access}
    nextlink = "https://api.github.com/search/commits"
                
    response = requests.get(nextlink, headers=myheaders,params=params)    
    headers = response.headers
    
    # If problems with the response, sleep until reset
    if response.status_code == 403:
        if headers['X-RateLimit-Remaining'] == 0:
            # Wait for rate limit to reset
            reset_ts = float(headers['X-RateLimit-Reset'])
            reset_time = datetime.datetime.fromtimestamp(reset_ts)
            now = datetime.datetime.now()
            sleep_time = (reset_time - now).total_seconds()
            sleep_time_secs1 = sleep_time*60
            if sleep_time_secs1 < 0:
                sleep_time_secs1 = sleep_time_secs1*(-1)
            # Sleep until the rate limit resets
            msg = 'Rate limit exceeded. Sleeping for {}s'.format(sleep_time_secs1)
            print(msg)
            time.sleep(sleep_time)
            response = requests.get(nextlink, headers=myheaders,params=params)
    
    # Determine how many pages of response
    if "Link" in headers:
        link = headers['Link']
        reflinks = analyzelinks(link)
        if "next" in reflinks and "last" in reflinks:
            last_page = reflinks["last"].split("&page=")[1]
            next_page = reflinks["next"].split("&page=")[1]
            pages_to_check = [reflinks["last"].split("&page=")[0]+'&page='+str(p) for p in range(int(next_page), int(last_page)+1)]
    else:
        last_page = 1
        next_page = 1
        pages_to_check = ['No pages to check']
    
    # See how many results for each word
    resp_cont = response.json()
    print(key)
    print('Number of pages: ', len(pages_to_check))
    
    # Go through pages of data
    for page in pages_to_check:
        page_link = page
        if page_link == 'No pages to check':
            response2 = response
        else:
            response2 = requests.get(page_link, headers=myheaders)         
            if response2.status_code == 403:
                headers2 = response2.headers
                if headers2['X-RateLimit-Remaining'] == 0:
                    # Wait for rate limit to reset
                    reset_ts2 = float(headers2['X-RateLimit-Reset'])
                    reset_time2 = datetime.datetime.fromtimestamp(reset_ts2)
                    now2 = datetime.datetime.now()
                    sleep_time2 = (reset_time2 - now2).total_seconds()
                    sleep_time_secs = sleep_time2*60
                    if sleep_time_secs < 0:
                        sleep_time_secs = sleep_time_secs*(-1)
                    if sleep_time_secs > 3600:
                        print('over 3600s ... breaking loop1')
                        break
                    # Sleep until the rate limit resets
                    message2 = 'Rate limit exceeded. Sleeping for {}s'.format(sleep_time_secs)
                    print(message2)
                    time.sleep(sleep_time2)
                    response2 = requests.get(page_link, headers=myheaders)                                      
            
        # If response2 has data, get content
        content = response2.json()       
        if 'items' in content:
            for k in range(0, len(content["items"])):
                repo = content["items"][k]["repository"]["html_url"]
                if repo not in commits:
                    c = {}
                    c["url"] = content["items"][k]["url"]
                    c["html_url"] = content["items"][k]["html_url"]
                    c["message"] = content["items"][k]["commit"]["message"]
                    c["sha"] = content["items"][k]["sha"]
                    c["keyword"] = key
                    commits[repo] = {}
                    commits[repo][content["items"][k]["sha"]] = c;
                else:
                    if content["items"][k]["sha"] not in commits[repo]:
                        c = {}
                        c["url"] = content["items"][k]["url"]
                        c["html_url"] = content["items"][k]["html_url"]
                        c["sha"] = content["items"][k]["sha"]
                        c["keyword"] = key
                        commits[repo][content["items"][k]["sha"]] = c;
        
            #save the commits that were found
            with open('all_commits.json', 'w') as outfile:
                json.dump(commits, outfile)              
            
        else:
            # If giving an error, try again after reset
            print('no items:', content['message'])
            pages_to_check.append(page)
            headers2 = response2.headers
            reset_ts2 = float(headers2['X-RateLimit-Reset'])
            reset_time2 = datetime.datetime.fromtimestamp(reset_ts2)
            now2 = datetime.datetime.now()
            sleep_time2 = (reset_time2 - now2).total_seconds()
            sleep_time_secs = sleep_time2*60
            if sleep_time_secs < 0:
                sleep_time_secs = sleep_time_secs*(-1)
            if sleep_time_secs > 3600:
                print('over 3600s ... breaking loop')
                break
            message3 = 'Rate limit exceeded. Sleeping for {}s'.format(sleep_time_secs)
            print(message3)
            time.sleep(sleep_time_secs)            


    #save the commits that were found
    print(str(len(commits)) + " commits found.\n")
    with open('all_commits.json', 'w') as outfile:
        json.dump(commits, outfile)

def analyzelinks(link):
    #get references to the next page of results
    link = link + ","
    reflinks = {}
    while "," in link:
        pos = link.find(",")
        text = link[:pos]
        rest = link[pos+1:]
        try:
            if "\"next\"" in text:
                text = text.split("<")[1]
                text = text.split(">;")[0]
                reflinks["next"]=text
            if "\"prev\"" in text:
                text = text.split("<")[1]
                text = text.split(">;")[0]
                reflinks["prev"]=text
            if "\"first\"" in text:
                text = text.split("<")[1]
                text = text.split(">;")[0]
                reflinks["first"]=text
            if "\"last\"" in text:
                text = text.split("<")[1]
                text = text.split(">;")[0]
                reflinks["last"]=text
        except IndexError as e:
            print(e)
            print("\n")
            print(text)
            print("\n\n")
            sys.exit()
        link = rest
    return(reflinks)

start = time.time()
print(start)

# Need a Github access token in this directory.
with open('Githubtoken', 'r') as accestoken:
    access = accestoken.readline().replace("\n","")

commits = {}

#load previously scraped commits
if os.path.isfile('all_commits.json'):
    with open('all_commits.json', 'r') as infile:
        commits = json.load(infile)

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

keywords_to_query = [i for k, v in keywords_dict.items() for i in v]

for k in keywords_to_query:
    searchforkeyword(k, commits, access)

end = time.time()
elapsed = (end-start)/60
print('time elapsed: ', elapsed)

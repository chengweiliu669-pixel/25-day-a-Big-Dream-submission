"""Reject commit identities outside the team's verified GitHub accounts."""
import json, re, subprocess, sys
ALLOWED = {
 '213363113+OscarZH2005030@users.noreply.github.com',
 '146118803+JaydenWang9@users.noreply.github.com',
 '343271245+chengweiliu669-pixel@users.noreply.github.com',
}
ref = sys.argv[1] if len(sys.argv)>1 else 'HEAD'
commits = subprocess.check_output(['git','rev-list',ref], text=True).splitlines()
seen=set(); failures=[]
for sha in commits:
 raw=subprocess.check_output(['git','cat-file','commit',sha], text=True)
 header,body=raw.split('\n\n',1)
 for line in header.splitlines():
  if line.startswith(('author ','committer ')):
   email=re.search(r'<([^>]+)>',line).group(1); seen.add(email)
   if email not in ALLOWED: failures.append({'commit':sha,'identity':email})
 for line in body.splitlines():
  if re.match(r'co-authored-by:',line,re.I):
   match=re.search(r'<([^>]+)>',line)
   if not match or match.group(1) not in ALLOWED:failures.append({'commit':sha,'coauthor':line})
result={'passed':not failures,'commit_count':len(commits),'identities':sorted(seen),'exact_three_present':seen==ALLOWED,'failures':failures}
print(json.dumps(result,indent=2))
sys.exit(0 if not failures and seen==ALLOWED else 1)

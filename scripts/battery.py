#!/usr/bin/env python3
"""Comprehensive quality+precision battery. usage: battery.py TAG
Greedy (temp 0) over a fixed set spanning math, code, knowledge, logic,
instruction-following, long-context recall and prose. Saves bat-<TAG>.json =
{id:{cat,prompt,answer,ok,toklp}} where toklp = mean logprob of the generated
tokens and top1 = list of top-1 token ids per generated position (for
position-aligned top-1 agreement vs native on items where answers match).
Auto-scores objective items; open items saved for judging."""
import json, os, sys, re, urllib.request

BASE = os.environ.get("BASE", "http://10.100.10.1:8000")
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash")
TAG = sys.argv[1]

def nums(t): return [float(x) for x in re.findall(r"-?\d[\d,]*\.?\d*", t.replace(",","")) ]
def last_num_eq(v, tol=0.05): return lambda t: bool(nums(t)) and abs(nums(t)[-1]-v) <= tol
def has(*s): return lambda t: all(x.lower() in t.lower() for x in s)
def anyof(*s): return lambda t: any(x.lower() in t.lower() for x in s)

# cat, id, prompt, max_tokens, checker
B = [
 ("math","m_avg","A train goes 360 km in 4 h then 150 km in 2.5 h. Average speed km/h for the whole trip? End with just the number.",400,last_num_eq(78.46,0.1)),
 ("math","m_change","Pens are 3 for $2. Maria buys 18 and pays $20. Change? End with just the dollar amount.",400,last_num_eq(8.0)),
 ("math","m_compound","$1000 at 10% annual interest compounded yearly for 3 years. Final amount? End with just the number.",500,last_num_eq(1331.0,1)),
 ("math","m_work","Pipe A fills a tank in 6 h, pipe B in 4 h. Both open, how many hours to fill? End with just the number (decimal ok).",500,last_num_eq(2.4,0.05)),
 ("math","m_prob","Two fair dice are rolled. Probability the sum is 7? Give it as a fraction in lowest terms.",300,anyof("1/6","6/36")),
 ("math","m_seq","What is the 10th term of the arithmetic sequence starting 4, 7, 10, ...? End with just the number.",300,last_num_eq(31.0)),
 ("math","m_pct","A shirt is marked down 20% then a further 25% off the reduced price. Overall percent discount? End with just the number.",400,last_num_eq(40.0)),
 ("reason","r_ages","Tom is twice as old as Sara was when Tom was as old as Sara is now. Sara is 30. How old is Tom? End with just the number.",500,last_num_eq(40.0)),
 ("reason","r_syll","All bloops are razzies. All razzies are lazzies. Can we conclude all bloops are lazzies? Yes or no, one reason.",200,has("yes")),
 ("reason","r_river","A farmer must cross a river with a wolf, a goat, and a cabbage; the boat holds the farmer plus one item. What does he take first to avoid any loss? Name the item.",300,has("goat")),
 ("reason","r_cal","If today is Wednesday, what day of the week is it 100 days from now? End with just the weekday name.",400,has("friday")),
 ("reason","r_clock","How many times do the hour and minute hands of a clock overlap in 24 hours? End with just the number.",400,last_num_eq(22.0)),
 ("know","k_elem","Chemical element with atomic number 74: name and symbol.",120,has("tungsten","w")),
 ("know","k_cap","Capital city of Australia?",60,has("canberra")),
 ("know","k_light","Speed of light in vacuum in m/s to 3 significant figures?",120,anyof("3.00","2.998","299792458","299,792,458")),
 ("know","k_planet","Which planet has the most moons as of 2025, and is it Jupiter or Saturn?",120,has("saturn")),
 ("know","k_author","Who wrote the novel 'One Hundred Years of Solitude'?",80,has("garc","marqu")),
 ("know","k_prime","Is 91 a prime number? Answer yes or no and give its factorization if not.",150,lambda t: ("no" in t.lower()) and ("7" in t and "13" in t)),
 ("code","c_palin","Write a Python function is_palindrome(s) ignoring case and non-alphanumerics. Only the code.",400,has("def is_palindrome")),
 ("code","c_fib","Write a Python function fib(n) returning the nth Fibonacci number (fib(0)=0, fib(1)=1), iterative. Only the code.",400,has("def fib")),
 ("code","c_fizz","Write Python that prints FizzBuzz for 1..15. Only the code.",400,has("fizz","buzz","%")),
 ("code","c_sql","Write a SQL query selecting the name and salary of the top 3 highest-paid employees from a table 'employees(name, salary)'.",300,lambda t: ("select" in t.lower()) and ("order by" in t.lower()) and ("limit 3" in t.lower() or "top 3" in t.lower() or "top(3)" in t.lower())),
 ("instr","i_colors","Reply with exactly three words naming primary colors of light, each capitalized, single spaces, nothing else.",40,lambda t: len(re.sub(r"[^A-Za-z ]","",t).split())==3),
 ("instr","i_json","Return ONLY valid minified JSON: an object with keys name (string 'Ada') and year (number 1815). No prose.",80,lambda t: (lambda m:(m and json.loads(m.group(0)).get("name")=="Ada" and json.loads(m.group(0)).get("year")==1815))(re.search(r"\{.*\}", t, re.S)) if re.search(r"\{.*\}", t, re.S) else False),
 ("instr","i_count","Write a sentence about the ocean that contains exactly five words. Output only the sentence.",40,lambda t: len(re.sub(r"[^A-Za-z ]","",t).split())==5),
 ("instr","i_nocomma","List three fruits separated by semicolons, no commas anywhere.",60,lambda t: (";" in t) and ("," not in t)),
 ("prose","p_autumn","Write exactly two sentences describing an autumn morning in a quiet town. Vivid but not purple.",200,None),
 ("prose","p_explain","Explain in 3 sentences, for a 12-year-old, why the sky is blue.",250,None),
 ("prose","p_haiku","Write a haiku about a lighthouse. Three lines, 5-7-5 syllables.",80,None),
 ("longctx","l_recall", None, 60, has("7431-q")),  # prompt built below
]

# long-context recall item: planted fact ~8K tokens in
fill = ("The ledger recorded each shipment of copper and salt across the northern passes. " * 400)
L_PROMPT = ("Read the archive below and answer the question at the end.\n\n" + fill[:4000] +
            "\nNOTE: the warehouse override key is 7431-Q. \n" + fill[:4000] +
            "\n\nQuestion: What is the warehouse override key stated in the note? Answer with the key only.")

out = {}
for cat, pid, prompt, mt, chk in B:
    if pid == "l_recall": prompt = L_PROMPT
    body = {"model": MODEL, "messages":[{"role":"user","content":prompt}], "max_tokens": mt,
            "temperature": 0, "seed": 0, "logprobs": True, "top_logprobs": 1}
    req = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    toklp, top1 = None, []
    try:
        r = urllib.request.urlopen(req, timeout=600); j = json.load(r)
        ch = j["choices"][0]; txt = ch["message"]["content"]
        lp = (ch.get("logprobs") or {}).get("content") or []
        if lp:
            vals = [e["logprob"] for e in lp if e.get("logprob") is not None]
            toklp = round(sum(vals)/len(vals), 4) if vals else None
            top1 = [e["token"] for e in lp]
    except Exception as e:
        txt = f"<ERROR {e}>"
    ok = None
    if chk is not None:
        try: ok = bool(chk(txt))
        except Exception: ok = False
    out[pid] = {"cat":cat, "prompt":(prompt if pid!="l_recall" else "<long-context recall, key 7431-Q>"),
                "answer":txt, "ok":ok, "toklp":toklp, "top1":top1}
    mark = {True:"PASS",False:"FAIL",None:"open"}[ok]
    print(f"[{mark}] {cat:<7} {pid:<10} {str(txt)[:80].replace(chr(10),' ')}")

scored = [(v["cat"],v["ok"]) for v in out.values() if v["ok"] is not None]
print(f"OBJECTIVE {sum(1 for _,x in scored if x)}/{len(scored)}")
bycat={}
for c,x in scored: bycat.setdefault(c,[0,0]); bycat[c][1]+=1; bycat[c][0]+= 1 if x else 0
print("by cat: "+", ".join(f"{c} {a}/{n}" for c,(a,n) in sorted(bycat.items())))
json.dump(out, open(f"bat-{TAG}.json","w"), indent=1)
print(f"saved bat-{TAG}.json")

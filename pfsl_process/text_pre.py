import re
import pandas as pd
import random
from functools import partial
import json

with open('/data/qingyuyang/prot_design/ProTrek/faiss_index/SwissProt/ProTrek_650M_UniRef50/text/subsections/Enzyme_commission_number_ids.tsv', 'r') as f:
    ec_database = [line.split(': ') for line in f.readlines()]

ec_dict = {line[0]: line[1] for line in ec_database}

gene_name_formats = [
    'The ORF Name <> identifies the gene responsible for this protein.',
    'The Ordered Locus Name <> indicates the gene that codes for this protein.',
    'The nucleotide sequence <> corresponds to the synthesis of this protein.'
]

# Protein names
def protein_name(text: str):
    # 正则匹配规则：
    # (.*?) ：非贪婪匹配开头文本（核心要提取的部分）
    # (?:\s*\([^)]*\))+ ：匹配1个或多个 "(...)" 组（前面可带空格）
    # (?:\s*\[[^]]*\])? ：匹配0个或1个 "[...]" 组（前面可带空格）
    # $ ：匹配字符串结尾，确保后缀是最后部分
    pattern = r'^(.*?)(?:\s*\([^)]*\))+(?:\s*\[[^]]*\])?$'
    match = re.match(pattern, text.strip())  # 先去除首尾空格，避免干扰
    if match:
        return match.group(1).strip()  # 去除开头文本的首尾空格
    # 处理只有方括号组的情况
    pattern_bracket = r'^(.*?)(?:\s*\[[^]]*\])$'
    match_bracket = re.match(pattern_bracket, text.strip())
    if match_bracket:
        return match_bracket.group(1).strip()
    # 无括号组的情况，返回原文本（去除首尾空格）
    return text.strip()

# Gene Names
def gene_names(text: str):
    return random.choice(gene_name_formats).replace('<>', text.split()[0])

# Activity regulation, Developmental stage, Induction, 
# Allergenic Properties (2, discard), Biotechnological use (5, discard),
# Domain [CC], Pathway, Caution, Function [CC], Miscellaneous [CC],
# Pharmaceutical use (5, discard), Polymorphism, Post-translational modification,
# Sequence similarities, Subunit structure, Tissue specificity
def general_process(item_name, raw_text: str | None):
    if type(raw_text) != str:
        return ''
    
    texts = raw_text[len(item_name) + 2:].split(f'; {item_name}: ') #
    # print(texts)
    results = []

    for text in texts:
        # text = ': '.join(text.split(': ')[1:])
        # print(text)
        text = text.split('. ')
        if text[-1].startswith('{'):
            text.pop()
        text = '. '.join(text)
        if not text.endswith('.'):
            text += '.'
        results.append(text)
    
    return ' '.join(results)

def cofactor(raw_text: str | None):
    if type(raw_text) != str:
        return ''
    
    results = []
    texts = raw_text[10:].split('; ')

    for text in texts:
        if text.startswith('Name='):
            results.append(f'Without {text[5:]} acting as a cofactor, this protein\'s function is compromised.')
        if text.startswith('Note='):
            results.append(text[5:].split(' {')[0])
    
    return ' '.join(results)

def proteome(raw_text: str):
    name = raw_text.split(': ')[1]
    if name == 'Unplaced':
        return ''
    return f'A proteome, containing {name}, is composed of this protein.'

# RNA Editing (4, discard)

def disease(raw_text: str | None):
    if type(raw_text) != str:
        return ''
    
    # raw_text.replace('Note=', '')
    texts = raw_text[len('DISEASE') + 2:].split(f'; DISEASE: ') #
    results = []

    for t in texts:
        t = t.split('Note=')[0].split(' [')[0].strip()
        if not t:
            continue
        results.append(f'{t} is the disease tied to genetic variations in this protein.')
    
    return ' '.join(results) if results else ''

def remove_braces_content(s):
    # 正则解释：
    # \s 匹配前面的一个空格（严格单个空格）
    # \{ 匹配左花括号
    # .*? 非贪婪匹配{}内任意字符（含换行）
    # \} 匹配右花括号
    # re.DOTALL 让.匹配换行符
    return re.sub(r'\s\{.*?\}', '', s, flags=re.DOTALL)

def rearrange_string(text):
    # 正则解释：
    # (.*?)          捕获[]前的所有文本（非贪婪，避免漏内容）
    # \[(.*?)\]:\s+  捕获[]内的内容，匹配"]: "（\s+适配多个空格）
    # (.*)           捕获[]: 后的剩余文本
    pattern = r'(.*?)\[(.*?)\]:\s+(.*)'
    match = re.search(pattern, text)
    
    if not match:
        return text  # 无匹配返回原字符串
    
    # 提取三部分内容：[]前的文本、[]内的文本、[]后的文本
    prefix = match.group(1).strip()    # []前的内容（如"The protein can be spotted at"）
    bracket_content = match.group(2).strip()  # []内的内容（如"Isoform 2"）
    suffix = match.group(3).strip()    # []后的内容（如"Cell membrane."）
    
    # 处理prefix首字母小写（重组后不再是句首）
    if prefix and prefix[0].isupper():
        prefix = prefix[0].lower() + prefix[1:]
    
    # 重组句子：[]内内容 + of + []前文本 + []后文本
    new_sentence = f"{bracket_content} of {prefix} {suffix}"
    
    # 清理多余空格（避免拼接后出现多个空格）
    new_sentence = re.sub(r'\s+', ' ', new_sentence)
    
    return new_sentence

location_words = ['local', 'Local', 'Colocal', 'colocal', 'In', '(By similarity)', # 'in', 
                  'Secreted', 'secreted', 'Associate', 'associate', 'Found', 'found',
                  'locat', 'Locat', 'Detect', 'detect', 'Recruit', 'recruit',
                  'predominantly', 'Predominantly']

def subcellular_location(raw_text: str | None):
    if type(raw_text) != str:
        return ''
    
    raw_text = raw_text[len('SUBCELLULAR LOCATION') + 2:]
    raw_text = remove_braces_content(raw_text)
    raw_text = raw_text.replace('; SUBCELLULAR LOCATION: ', ' ')
    # return raw_text
    
    texts = raw_text.split('.')
    phrase_res = []
    sentence_res = []
    
    for t in texts:
        if not t or t == 'Secreted':
            continue
        t = t.strip()
        t = t.replace('Note=', '')
        # print(t)

        if_sentence = False
        for w in location_words:
            if w in t: #.split(';')[0]
                # print(w)
                sentence_res.append(t + '.') #
                if_sentence = True
                break
        if not if_sentence:
            res = 'The protein can be spotted at ' + t + '.'
            if '[' in res:
                res = rearrange_string(res)
            phrase_res.append(res) #
    
    if phrase_res:
        phrase_cat = ' '.join([i for i in phrase_res])
        if not phrase_cat.endswith('.'):
            phrase_cat += '.'
    else:
        phrase_cat = ''
    
    if sentence_res:
        sentence_cat = ' '.join(sentence_res)
    else:
        sentence_cat = ''
        
    return phrase_cat + ' ' + sentence_cat if phrase_cat and sentence_cat else phrase_cat + sentence_cat

def catalytic_activity(raw_text: str | None):
    if type(raw_text) != str:
        return ''
    
    texts = raw_text[len('CATALYTIC ACTIVITY') + 2:].split(f'; CATALYTIC ACTIVITY: ') #
    # print(texts)
    results = []

    for text in texts:
        chunks = text.split('; ')

        for chunk in chunks:
            if chunk.startswith('Reaction='):
                results.append(chunk[9:])
    
    res_texts = ['The reaction: {} undergoes catalysis due to the presence of this protein.'.format(r) for r in results]
    return ' '.join(res_texts)

def ec_convert(raw_text: str | None):
    if type(raw_text) != str:
        return ''
    
    results = []
    texts = raw_text.split('; ')
    for text in texts:
        results.append(ec_dict.get('EC ' + text, ''))

    return ' '.join(results)


if __name__ == '__main__':
    function_map = pd.read_csv('../data/prot_func/idmapping_reviewed_true_2026_01_09.tsv', sep='\t')
    function_map = function_map.drop(['Allergenic Properties', 'Biotechnological use', 'Pharmaceutical use', 'RNA Editing'], axis=1)

    function_map['Protein names'] = function_map['Protein names'].map(lambda x: f'This protein is identified as {protein_name(x)}.')
    function_map['Gene Names'] = function_map['Gene Names'].map(gene_names)
    # function_map['Organism'] = function_map['Organism'].map(lambda x: f'The organism\'s name {x} is linked to the protein sequence.')
    function_map['Activity regulation'] = function_map['Activity regulation'].map(partial(general_process, 'ACTIVITY REGULATION'))
    function_map['Developmental stage'] = function_map['Developmental stage'].map(partial(general_process, 'DEVELOPMENTAL STAGE'))
    function_map['Induction'] = function_map['Induction'].map(partial(general_process, 'INDUCTION'))
    function_map['Gene Ontology (biological process)'] = function_map['Gene Ontology (biological process)'].map(lambda x: f"The GO term for this protein involving biological process incorporates {x}." if type(x) == str else '')
    function_map['Gene Ontology (cellular component)'] = function_map['Gene Ontology (cellular component)'].map(lambda x: f"The GO term for this protein involving cellular component incorporates {x}." if type(x) == str else '')
    function_map['Gene Ontology (molecular function)'] = function_map['Gene Ontology (molecular function)'].map(lambda x: f"The GO term for this protein involving molecular function incorporates {x}." if type(x) == str else '')
    function_map['Involvement in disease'] = function_map['Involvement in disease'].map(disease)
    function_map['Subcellular location [CC]'] = function_map['Subcellular location [CC]'].map(subcellular_location)
    function_map['Domain [CC]'] = function_map['Domain [CC]'].map(partial(general_process, 'DOMAIN'))
    function_map['Pathway'] = function_map['Pathway'].map(partial(general_process, 'PATHWAY'))
    function_map['Cofactor'] = function_map['Cofactor'].map(cofactor)
    function_map['Catalytic activity'] = function_map['Catalytic activity'].map(catalytic_activity)
    function_map['EC number'] = function_map['EC number'].map(ec_convert)
    function_map['Caution'] = function_map['Caution'].map(partial(general_process, 'CAUTION'))
    function_map['Function [CC]'] = function_map['Function [CC]'].map(partial(general_process, 'FUNCTION'))
    function_map['Miscellaneous [CC]'] = function_map['Miscellaneous [CC]'].map(partial(general_process, 'MISCELLANEOUS'))
    function_map['Polymorphism'] = function_map['Polymorphism'].map(partial(general_process, 'POLYMORPHISM'))
    function_map['Post-translational modification'] = function_map['Post-translational modification'].map(partial(general_process, 'PTM'))
    function_map['Proteomes'] = function_map['Proteomes'].map(proteome)
    function_map['Sequence similarities'] = function_map['Sequence similarities'].map(partial(general_process, 'SIMILARITY'))
    function_map['Subunit structure'] = function_map['Subunit structure'].map(partial(general_process, 'SUBUNIT'))
    function_map['Tissue specificity'] = function_map['Tissue specificity'].map(partial(general_process, 'TISSUE SPECIFICITY'))

    label_list = [
        'Protein names',
        'Gene Names',
        'Activity regulation',
        'Developmental stage',
        'Induction',
        'Gene Ontology (biological process)',
        'Gene Ontology (cellular component)',
        'Gene Ontology (molecular function)',
        'Involvement in disease',
        'Subcellular location [CC]',
        'Domain [CC]',
        'Pathway',
        'Cofactor',
        'Catalytic activity',
        'EC number',
        'Caution',
        'Function [CC]',
        'Miscellaneous [CC]',
        'Polymorphism',
        'Post-translational modification',
        'Proteomes',
        'Sequence similarities',
        'Subunit structure',
        'Tissue specificity'
    ]
    function_map['All'] = function_map[label_list].apply(lambda row: row.tolist(), axis=1)

    res_dict = dict(zip(function_map['From'], function_map['All']))
    with open('../data/prot_func/all_text.txt', 'w') as f:
        json.dump(res_dict, f, indent=4)

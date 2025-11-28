#!/usr/bin/env python3
"""Simple NYC Mayor Knowledge Graph Builder - ~100 lines"""
import re
from datetime import date
from io import StringIO
import pandas as pd
import requests
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import FOAF, RDF, XSD

START = 2000
END = 2025

# Configuration
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_mayors_of_New_York_City"
DBPEDIA = "https://dbpedia.org/sparql"
TIME_START, TIME_END = date(START, 1, 1), date(END, 12, 31)
OUTPUT = f"mayors_{TIME_START.year}_{TIME_END.year}.ttl"
NYC = Namespace("http://example.org/nyc/mayors/")

def slugify(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name.strip()).strip('_')

def parse_date(text):
    text = str(text).strip()
    if text.lower() in {'present', 'incumbent', 'ongoing', ''}:
        return None
    match = re.search(r'(\d{4})', text)
    return date(int(match.group(1)), 1, 1) if match else None

def fetch_mayors():
    """Scrape Wikipedia tables"""
    response = requests.get(WIKI_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
    tables = pd.read_html(StringIO(response.text))
    all_data = []
    
    for tbl in tables:
        if isinstance(tbl.columns, pd.MultiIndex):
            tbl.columns = [' '.join(str(c).strip() for c in col if str(c).strip()) for col in tbl.columns]
        tbl = tbl.rename(columns={c: 'No' if re.search(r'^No', str(c), re.I) else 
                                  'Name' if re.search(r'name', str(c), re.I) else 
                                  'Term' if re.search(r'term', str(c), re.I) else c 
                                  for c in tbl.columns})
        if 'No' in tbl.columns and 'Name' in tbl.columns:
            all_data.append(tbl)
    
    df = pd.concat(all_data, ignore_index=True)
    mayors = []
    
    for _, row in df.iterrows():
        if not re.match(r'^\d+$', str(row.get('No', '')).strip()):
            continue
        name_cell = row.get('Name', '')
        if pd.isna(name_cell):
            continue
        name = re.sub(r'[\d†]+$', '', str(name_cell).split('(')[0]).strip()
        term = str(row.get('Term', ''))
        
        # Try multi-year pattern first
        dates = re.findall(r'(\d{4})\s*[–—-]\s*(\d{4}|present)', term)
        if not dates:
            # Try single year or full dates
            years = re.findall(r'\b(\d{4})\b', term)
            if years:
                dates = [(years[0], years[-1] if len(years) > 1 else years[0])]
        
        for start_year, end_year in dates:
            start = parse_date(start_year)
            end = parse_date(end_year)
            # Include if term overlaps with TIME_START to TIME_END range
            if start and (not end or end >= TIME_START) and start <= TIME_END:
                mayors.append({'name': name, 'slug': slugify(name), 'start': start, 'end': end})
                print(f"{row['No']} {name}")
    
    return mayors

def fetch_parties(mayors):
    """Get party and birthplace info from DBpedia"""
    slugs = ' '.join(f"dbr:{m['slug']}" for m in mayors)
    query = f"""PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX dbp: <http://dbpedia.org/property/>
SELECT ?mayor ?party ?partyLabel ?birthPlace ?birthPlaceLabel WHERE {{
  VALUES ?mayor {{ {slugs} }}
  OPTIONAL {{ ?mayor dbo:party ?party . ?party foaf:name ?partyLabel }}
  OPTIONAL {{ 
    ?mayor dbo:birthPlace ?birthPlace . 
    ?birthPlace rdfs:label ?birthPlaceLabel .
    FILTER(LANG(?birthPlaceLabel) = "en")
  }}
}}"""
    
    r = requests.get(DBPEDIA, params={'query': query, 'format': 'json'}, timeout=60)
    bindings = r.json()['results']['bindings']
    
    data = {}
    for row in bindings:
        uri = row['mayor']['value']
        if uri not in data:
            data[uri] = {}
        if 'party' in row:
            data[uri]['party'] = {
                'uri': row['party']['value'], 
                'label': row.get('partyLabel', {}).get('value')
            }
        if 'birthPlace' in row:
            data[uri]['birthPlace'] = {
                'uri': row['birthPlace']['value'],
                'label': row.get('birthPlaceLabel', {}).get('value')
            }
    return data

def build_graph(mayors, dbpedia_data):
    """Create RDF graph"""
    g = Graph()
    g.bind('nyc', NYC)
    g.bind('foaf', FOAF)
    
    for m in mayors:
        node = NYC[m['slug']]
        g.add((node, RDF.type, NYC.Mayor))
        g.add((node, FOAF.name, Literal(m['name'])))
        
        # Term with duration
        term = NYC[f"term_{m['start'].isoformat()}"]
        g.add((term, RDF.type, NYC.Term))
        g.add((term, NYC.termStart, Literal(m['start'].isoformat(), datatype=XSD.date)))

        if m['end']:
            g.add((term, NYC.termEnd, Literal(m['end'].isoformat(), datatype=XSD.date)))
            duration = (m['end'] - m['start']).days
            g.add((term, NYC.termDuration, Literal(duration, datatype=XSD.integer)))
        g.add((node, NYC.hasTerm, term))
        
        # DBpedia data
        dbp_uri = f"http://dbpedia.org/resource/{m['slug']}"
        if dbp_uri in dbpedia_data:
            data = dbpedia_data[dbp_uri]
            
            # Party
            if 'party' in data:
                p = data['party']
                party_node = NYC[f"party_{slugify(p['label'])}"]
                g.add((party_node, RDF.type, NYC.PoliticalParty))
                g.add((party_node, FOAF.name, Literal(p['label'])))
                g.add((node, NYC.memberOf, party_node))
            
            # Birth place
            if 'birthPlace' in data:
                bp = data['birthPlace']
                place_node = NYC[f"place_{slugify(bp['label'])}"]
                g.add((place_node, RDF.type, NYC.Location))
                g.add((place_node, FOAF.name, Literal(bp['label'])))
                g.add((node, NYC.birthPlace, place_node))
    
    # Succession
    for i in range(len(mayors) - 1):
        curr = NYC[mayors[i]['slug']]
        nxt = NYC[mayors[i + 1]['slug']]
        g.add((nxt, NYC.precededBy, curr))
        g.add((curr, NYC.succeededBy, nxt))
    
    return g

if __name__ == '__main__':
    print('Fetching mayors...')
    mayors = fetch_mayors()
    print(f'Fetching DBpedia data for {len(mayors)} mayors...')
    dbpedia_data = fetch_parties(mayors)
    print('Building graph...')
    graph = build_graph(mayors, dbpedia_data)
    graph.serialize(OUTPUT, format='turtle')
    print(f'Saved to {OUTPUT}')

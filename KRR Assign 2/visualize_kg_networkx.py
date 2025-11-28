#!/usr/bin/env python3
"""Visualize NYC Mayors Knowledge Graph using NetworkX and Matplotlib (matching class example)."""
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, Namespace, RDF
from rdflib.namespace import FOAF

START = 1973
END = 2025

PATH = f"mayors_{START}_{END}.ttl"

NYC = Namespace("http://example.org/nyc/mayors/")

def load_rdf_to_networkx(ttl_path: Path) -> nx.MultiDiGraph:
    """Load TTL into RDFLib, then convert to NetworkX MultiDiGraph with enhanced relationships."""
    rdf_g = Graph()
    rdf_g.parse(ttl_path, format="turtle")
    
    G = nx.MultiDiGraph()
    
    # Collect mayor data first
    mayor_data = {}
    for mayor in rdf_g.subjects(RDF.type, NYC.Mayor):
        name = str(next(rdf_g.objects(mayor, FOAF.name), "Unknown"))
        term_count = sum(1 for _ in rdf_g.objects(mayor, NYC.hasTerm))
        parties = [str(next(rdf_g.objects(p, FOAF.name), "")) 
                   for p in rdf_g.objects(mayor, NYC.memberOf)]
        primary_party = parties[0] if parties else "Unknown"
        
        # Get birth place
        birthplace = None
        for bp in rdf_g.objects(mayor, NYC.birthPlace):
            birthplace = str(next(rdf_g.objects(bp, FOAF.name), ""))
            break
        
        # Get term dates
        years = []
        for term in rdf_g.objects(mayor, NYC.hasTerm):
            start = next(rdf_g.objects(term, NYC.termStart), None)
            end = next(rdf_g.objects(term, NYC.termEnd), None)
            if start:
                years.append(str(start)[:4])
            if end:
                years.append(str(end)[:4])
        
        duration = f"({min(years)}–{max(years)})" if years else ""
        display_name = f"{name}\n{duration}" if duration else name
        
        mayor_data[str(mayor)] = {
            'name': name,
            'display_name': display_name,
            'term_count': term_count,
            'parties': parties,
            'primary_party': primary_party,
            'birthplace': birthplace
        }
        
        # Add mayor nodes
        G.add_node(
            display_name,
            uri=str(mayor),
            frequency=term_count,
            size=max(300, term_count * 200),
            party=primary_party,
            node_type='mayor',
            original_name=name,
            birthplace=birthplace
        )
    
    # Add party nodes and connect mayors to parties
    party_to_mayors = {}
    for uri, data in mayor_data.items():
        party = data['primary_party']
        if party != "Unknown":
            if party not in party_to_mayors:
                party_to_mayors[party] = []
                G.add_node(
                    f"Party: {party}",
                    size=500,
                    node_type='party',
                    party=party
                )
            party_to_mayors[party].append(data['display_name'])
            G.add_edge(data['display_name'], f"Party: {party}", 
                      label="memberOf", relationship="memberOf")
    
    # Add birthplace nodes and connect mayors to birthplaces
    birthplace_to_mayors = {}
    for uri, data in mayor_data.items():
        birthplace = data.get('birthplace')
        if birthplace:
            if birthplace not in birthplace_to_mayors:
                birthplace_to_mayors[birthplace] = []
                G.add_node(
                    f"📍 {birthplace}",
                    size=400,
                    node_type='location',
                    location=birthplace
                )
            birthplace_to_mayors[birthplace].append(data['display_name'])
            G.add_edge(data['display_name'], f"📍 {birthplace}", 
                      label="birthPlace", relationship="birthPlace")
    
    # Add re-election indicator
    multi_term_mayors = [data['display_name'] for data in mayor_data.values() if data['term_count'] > 1]
    if multi_term_mayors:
        G.add_node(
            "Re-elected",
            size=600,
            node_type='indicator',
            party='Special'
        )
        for mayor_name in multi_term_mayors:
            G.add_edge(mayor_name, "Re-elected", 
                      label="hasTerm (×2+)", relationship="reelected")
    
    # Add succession edges - need to map original names to display names
    # Only use precededBy to avoid duplicates (since both precededBy and succeededBy exist in TTL)
    name_to_display = {data['name']: data['display_name'] for data in mayor_data.values()}
    added_succession = set()
    
    for mayor in rdf_g.subjects(RDF.type, NYC.Mayor):
        name1 = str(next(rdf_g.objects(mayor, FOAF.name), "Unknown"))
        display1 = name_to_display.get(name1, name1)
        for pred in rdf_g.objects(mayor, NYC.precededBy):
            name2 = str(next(rdf_g.objects(pred, FOAF.name), "Unknown"))
            display2 = name_to_display.get(name2, name2)
            edge_key = (display2, display1)
            if display1 in G and display2 in G and edge_key not in added_succession:
                G.add_edge(display2, display1, label="precededBy", relationship="succession")
                added_succession.add(edge_key)
    
    # Add same-party connections (limited to avoid clutter)
    for party, mayors in party_to_mayors.items():
        if len(mayors) > 1:
            for i, mayor1 in enumerate(mayors):
                for mayor2 in mayors[i+1:min(i+3, len(mayors))]:
                    if mayor1 in G and mayor2 in G:
                        G.add_edge(mayor1, mayor2, 
                                  label="sameParty", relationship="sameParty")
    
    return G

def visualize_graph(graph: nx.MultiDiGraph, figsize=(16, 12), output_path: Path = None):
    """Visualize the knowledge graph using NetworkX and Matplotlib with colored node types."""
    plt.figure(figsize=figsize)
    
    # Separate nodes by type
    mayor_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'mayor']
    party_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'party']
    location_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'location']
    indicator_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'indicator']
    
    # Create custom layout: special nodes in center, mayors in circle
    pos = {}
    
    import math
    
    # Get mayors sorted chronologically
    mayor_names = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'mayor']
    
    # Sort mayors chronologically by extracting year from display name
    def get_start_year(name):
        import re
        match = re.search(r'\((\d{4})', name)
        return int(match.group(1)) if match else 9999
    
    sorted_mayors = sorted(mayor_names, key=get_start_year)
    
    # Position mayors in circle in chronological order
    if sorted_mayors:
        angle_step = 2 * math.pi / len(sorted_mayors)
        radius = 2.5  # Large outer circle for mayors
        for i, mayor in enumerate(sorted_mayors):
            angle = i * angle_step - math.pi/2  # Start at top
            pos[mayor] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Center positions for party, location, and indicator nodes
    center_nodes = party_nodes + location_nodes + indicator_nodes
    if center_nodes:
        angle_step = 2 * math.pi / max(len(center_nodes), 1)
        radius = 0.8  # Small inner circle for special nodes
        for i, node in enumerate(center_nodes):
            angle = i * angle_step
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Separate nodes by type for different colors
    mayor_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'mayor']
    party_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'party']
    location_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'location']
    indicator_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'indicator']
    
    # Draw mayor nodes (lightblue)
    if mayor_nodes:
        mayor_sizes = [graph.nodes[n].get('size', 300) for n in mayor_nodes]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=mayor_nodes,
            node_size=mayor_sizes,
            node_color='lightblue',
            alpha=0.7,
            edgecolors='black',
            linewidths=1
        )
    
    # Draw party nodes (lightgreen)
    if party_nodes:
        party_sizes = [graph.nodes[n].get('size', 500) for n in party_nodes]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=party_nodes,
            node_size=party_sizes,
            node_color='lightgreen',
            alpha=0.8,
            edgecolors='darkgreen',
            linewidths=2,
            node_shape='s'  # Square for parties
        )
    
    # Draw location nodes (lightcoral)
    if location_nodes:
        location_sizes = [graph.nodes[n].get('size', 400) for n in location_nodes]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=location_nodes,
            node_size=location_sizes,
            node_color='lightcoral',
            alpha=0.8,
            edgecolors='darkred',
            linewidths=2,
            node_shape='^'  # Triangle for locations
        )
    
    # Draw indicator nodes (lightyellow)
    if indicator_nodes:
        indicator_sizes = [graph.nodes[n].get('size', 600) for n in indicator_nodes]
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=indicator_nodes,
            node_size=indicator_sizes,
            node_color='lightyellow',
            alpha=0.8,
            edgecolors='orange',
            linewidths=2,
            node_shape='d'  # Diamond for indicators
        )
    
    # Draw different edge types with different styles
    succession_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relationship') == 'succession']
    member_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relationship') == 'memberOf']
    birthplace_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relationship') == 'birthPlace']
    reelected_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relationship') == 'reelected']
    same_party_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get('relationship') == 'sameParty']
    
    # Succession edges (solid gray with arrows)
    if succession_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=succession_edges,
            alpha=0.5,
            edge_color='darkgray',
            arrows=True,
            arrowsize=18,
            width=2,
            arrowstyle='-|>',
            connectionstyle="arc3,rad=0.0"
        )
    
    # Membership edges (dashed green)
    if member_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=member_edges,
            alpha=0.3,
            edge_color='green',
            arrows=False,
            width=1,
            style='dashed'
        )
    
    # Birthplace edges (dotted red)
    if birthplace_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=birthplace_edges,
            alpha=0.4,
            edge_color='red',
            arrows=False,
            width=1.5,
            style='dotted'
        )
    
    # Re-elected edges (dotted orange)
    if reelected_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=reelected_edges,
            alpha=0.4,
            edge_color='orange',
            arrows=False,
            width=1.5,
            style='dotted'
        )
    
    # Same-party edges (very light blue, thin)
    if same_party_edges:
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=same_party_edges,
            alpha=0.15,
            edge_color='blue',
            arrows=False,
            width=0.5
        )
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos,
        font_size=7,
        font_weight='bold',
        font_family='sans-serif'
    )
    
    # Draw edge labels for all relationships
    edge_labels = {}
    for u, v, d in graph.edges(data=True):
        label = d.get('label', '')
        rel = d.get('relationship', '')
        # Only show labels for succession edges to reduce clutter
        if label and rel == 'succession':
            edge_labels[(u, v)] = label
    
    if edge_labels:
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=6,
            font_color='darkred',
            font_weight='bold',
            alpha=0.9,
        )
    
    plt.title(f"NYC Mayors Knowledge Graph ({START}–{END})\nBlue=Mayors | Green=Parties | Red=Birthplaces | Yellow=Re-elected", 
              size=16, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()

def main():
    ttl_path = Path(PATH) 
    if not ttl_path.exists():
        print(f"Error: {ttl_path} not found")
        return
    
    print("Loading RDF graph and converting to NetworkX...")
    G = load_rdf_to_networkx(ttl_path)
    
    print(f"Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    print("Generating visualization...")
    output_path = Path(f"viz_output/nyc_mayors_networkx_{START}_{END}.png")
    output_path.parent.mkdir(exist_ok=True)
    
    visualize_graph(G, figsize=(18, 14), output_path=output_path)
    
    print("Done!")

if __name__ == "__main__":
    main()

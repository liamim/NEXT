from pathlib import Path
import networkx as nx
import yaml

G = nx.grid_2d_graph(21, 21)
filepaths = []
neighbors = []
locations = []
ids = []
for i, p in enumerate(Path('.').glob('cat_dog/*.png')):
    x, y = map(int, p.stem.split('_'))
    n = (x+10, y+10)
    neighbors.append(list(G.neighbors(n)))
    G.nodes[n]['i'] = i
    ids.append(i)
    locations.append((x, y))
    filepaths.append(p)

targets = [
    {'primary_description': f'http://localhost:8889/{p.name}',
     'primary_type': 'image',
     'alt_description': '',
     'alt_type': '',
     # 'neighbors_': ns,
     'location': loc,
     'neighbors': [G.nodes[n]['i'] for n in ns],
     'graph_id': graph_id,
     'node_id': i,
    }
    for p, i, ns, loc in zip(filepaths, ids, neighbors, locations) for graph_id in range(2)
]

print(yaml.safe_dump(targets))

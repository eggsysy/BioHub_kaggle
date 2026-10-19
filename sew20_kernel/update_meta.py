import json

with open('kernel-metadata.json', 'r') as f:
    meta = json.load(f)

meta['title'] = 'biohub-948-sew20'
if 'docker_image' in meta:
    del meta['docker_image']
if 'id_no' in meta:
    del meta['id_no']

with open('kernel-metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)


import json
from collections import defaultdict

def count_objects_by_category(json_data):

    # with open(json_path, 'r') as f:
    #     data = json.load(f)

    supercat_counts = defaultdict(int)
    category_counts = defaultdict(lambda: defaultdict(int))
    for item in data:
        super_cat = item["super-category"]
        cat = item["category"]
        supercat_counts[super_cat] += 1
        category_counts[super_cat][cat] += 1

    print("=== Object Counts by Super-Category and Category ===")
    for super_cat, total in supercat_counts.items():
        print(f"\nSuper-Category: {super_cat} (Total: {total})")
        for cat, count in category_counts[super_cat].items():
            print(f"  - Category: {cat} → {count} objects")        

'''
=== Object Counts by Super-Category and Category ===

Super-Category: Sofa (Total: 2701)
  - Category: armchair → 751 objects
  - Category: Three-Seat / Multi-seat Sofa → 852 objects
  - Category: Loveseat Sofa → 472 objects
  - Category: Lazy Sofa → 112 objects
  - Category: L-shaped Sofa → 245 objects
  - Category: Footstool / Sofastool / Bed End Stool / Stool → 83 objects
  - Category: Chaise Longue Sofa → 25 objects
  - Category: Three-Seat / Multi-person sofa → 129 objects
  - Category: U-shaped Sofa → 10 objects
  - Category: Two-seat Sofa → 22 objects

Super-Category: Chair (Total: 1775)
  - Category: Lounge Chair / Cafe Chair / Office Chair → 585 objects
  - Category: Dining Chair → 543 objects
  - Category: Barstool → 191 objects
  - Category: Dressing Chair → 75 objects
  - Category: Classic Chinese Chair → 16 objects
  - Category: Lounge Chair / Book-chair / Computer Chair → 325 objects
  - Category: Hanging Chair → 14 objects
  - Category: Folding chair → 26 objects

Super-Category: Lighting (Total: 1921)
  - Category: Pendant Lamp → 1152 objects
  - Category: Ceiling Lamp → 512 objects
  - Category: Floor Lamp → 252 objects
  - Category: Wall Lamp → 5 objects

Super-Category: Cabinet/Shelf/Desk (Total: 5725)
  - Category: Coffee Table → 684 objects
  - Category: Corner/Side Table → 718 objects
  - Category: Nightstand → 538 objects
  - Category: Bookcase / jewelry Armoire → 439 objects
  - Category: TV Stand → 529 objects
  - Category: Drawer Chest / Corner cabinet → 509 objects
  - Category: Shelf → 275 objects
  - Category: Wardrobe → 476 objects
  - Category: Sideboard / Side Cabinet / Console Table → 434 objects
  - Category: Children Cabinet → 352 objects
  - Category: Round End Table → 99 objects
  - Category: Wine Cabinet → 114 objects
  - Category: Sideboard / Side Cabinet / Console → 244 objects
  - Category: Shoe Cabinet → 83 objects
  - Category: Wine Cooler → 111 objects
  - Category: Tea Table → 120 objects

Super-Category: Table (Total: 1090)
  - Category: Dining Table → 517 objects
  - Category: Desk → 356 objects
  - Category: Dressing Table → 142 objects
  - Category: Sideboard / Side Cabinet / Console Table → 24 objects
  - Category: Bar → 47 objects
  - Category: None → 4 objects

Super-Category: Bed (Total: 1124)
  - Category: King-size Bed → 440 objects
  - Category: Bed Frame → 203 objects
  - Category: Single bed → 135 objects
  - Category: Kids Bed → 118 objects
  - Category: Bunk Bed → 66 objects
  - Category: Double Bed → 141 objects
  - Category: Couch Bed → 14 objects
  - Category: None → 7 objects

Super-Category: Pier/Stool (Total: 487)
  - Category: Footstool / Sofastool / Bed End Stool / Stool → 487 objects

Super-Category: Others (Total: 1740)
  - Category: None → 1740 objects

'''

def filter_by_supercategory(supercategory : list,
                            json_path = '/scratch/leuven/375/vsc37593/3D-FUTURE-model/3D-FUTURE-model/model_info.json', 
                            length = None):

    '''
    filter 3d future data by super-category
    return a dictionary of super-category and its items
    '''

    category_dict = defaultdict(list)
    with open(json_path, 'r') as f:
        data = json.load(f)

        # iterate through the json data
        # and filter by super-category
        for idx, item in enumerate(data):
            if (item['super-category'] in supercategory and
                item['category'] is not None and
                item['style'] is not None and
                item['theme'] is not None):
                category_dict[item['super-category']].append(item)
        
        result = defaultdict(list)
        total_selected = 0
        for super_cat, items in category_dict.items():
            if length is not None:
                if len(items) < length:
                    print(f"Warning: Super-category '{super_cat}' has only {len(items)} items (less than {length})")
                selected_items = items[:length]
            else:
                selected_items = items
            
            result[super_cat] = selected_items
            total_selected += len(selected_items)
        print(f"Total selected meshes: {total_selected}")
    return result



if __name__ == '__main__':

    model_json_path = '/scratch/leuven/375/vsc37593/3D-FUTURE-model/3D-FUTURE-model/model_info.json'
    #count_objects_by_category(model_json_path)
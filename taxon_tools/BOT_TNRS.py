import requests
import pandas as pd
import json

def iterate_taxon_resolve(taxon_frame):
    """iterate_taxon_resolve: uses process taxon resolve, to take taxonomic names with missing
        infraspecific rank, and re-process them with placeholder infraspecific rank var. and subsp."""

    results = process_taxon_resolve(taxon_frame)

    failed_results = results[results['overall_score'] <= .99].copy()

    if len(failed_results) > 0:

        failed_results['fullname'] = failed_results['fullname'].replace({' var. ': ' subsp. ',
                                                                         ' subsp. ': ' var. '}, regex=True)

        failed_results = failed_results[['CatalogNumber', 'fullname']]

        failed_results = process_taxon_resolve(failed_results)

        results = pd.concat([results, failed_results], ignore_index=True)

        new_index = results.groupby('CatalogNumber')['overall_score'].idxmax()

        results = results.loc[new_index]
    else:
        pass

    results = results.drop_duplicates(subset='CatalogNumber')

    return results

def process_taxon_resolve(taxon_frame):
    """process_taxon_resolve: uses TNRS or the taxonomic name resolution service
        to process, batches of taxonomic names, correct for spelling mistakes,
        and flag unrecognized or new taxa.
        args:
            taxon_frame: a dataframe of taxonomic names with barcodes, or unique numeric identifier
        returns:
            a dataframe containing spelling corrected taxonomic names, matched names ,and taxonomic authors.
            contains accuracy score from TNRS, for filtering based on quality of match.
    """

    url_tn = "https://tnrsapi.xyz/tnrs_api.php"

    unique_taxon = taxon_frame.drop_duplicates(subset='fullname', keep='first')

    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'charset': 'UTF-8'
    }

    data_json = unique_taxon.to_json(orient='records')

    sources = "wfo,wcvp"
    class_val = "wfo"
    mode = "resolve"
    match = "best"

    opts = {
        "sources": sources,
        "class": class_val,
        "mode": mode,
        "matches": match
    }

    input_json = json.dumps({"opts": opts, "data": json.loads(data_json)})

    results_json = requests.post(url_tn, headers=headers, data=input_json.encode('utf-8'))

    results_raw = results_json.json()

    results = pd.DataFrame(results_raw)

    # Replacing empty strings with NaN
    results['Overall_score'] = results['Overall_score'].replace('', 0)

    # Converting to float, handling non-convertible values by setting errors='coerce'
    results['match.score'] = pd.to_numeric(results['Overall_score'], errors='coerce').round(2).astype(str)

    # Selecting specific columns
    results = results[['Name_submitted', 'Overall_score', 'Name_matched', 'Taxonomic_status',
                       'Accepted_name', 'Unmatched_terms', 'Canonical_author', 'Accepted_name_author']]

    results = results.rename(columns={'Name_submitted': 'fullname',
                                      'Name_matched': 'name_matched',
                                      'Taxonomic_status': 'taxonomic_status',
                                      'Accepted_name': 'accepted_name',
                                      'Unmatched_terms': 'unmatched_terms',
                                      'Overall_score': 'overall_score',
                                      'Accepted_name_author': 'accepted_name_author',
                                      'Canonical_author': 'matched_name_author'})

    # Left join with taxon_frame
    results = pd.merge(taxon_frame, results, on='fullname', how='left')

    # Selecting specific columns
    results = results[['fullname', 'name_matched', 'matched_name_author', 'accepted_name_author', 'overall_score',
                       'unmatched_terms', 'CatalogNumber']]

    results['overall_score'] = pd.to_numeric(results['overall_score'], errors='coerce')

    return results

# taxon_frame = pd.DataFrame({"CatalogNumber": ['123456'], "fullname": "Fissidens submarginatus"})
#
# pd.set_option('display.max_columns', None)
#
# results = iterate_taxon_resolve(taxon_frame=taxon_frame)
#
# print(results)

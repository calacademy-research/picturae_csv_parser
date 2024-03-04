"""tests for the taxon_concat and taxon_check_real function (using tnrs)"""
import pandas as pd
import unittest
from tests.pic_csv_test_class import AltCsvCreatePicturae
from tests.testing_tools import TestingTools
from taxon_tools.BOT_TNRS import iterate_taxon_resolve

class ConcatTaxonTests(unittest.TestCase, TestingTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.md5_hash = self.generate_random_md5()
    def setUp(self):
        """creates fake taxon columns in
           dummy dataset to test out taxon_concat string output"""
        self.test_csv_create_picturae = AltCsvCreatePicturae(date_string=self.md5_hash, logging_level='DEBUG')

        # jose Gonzalez is a real agent,
        # to make sure true matches are not added to list.
        # abies balsamea given incorrect placeholder rank "subsp." instead of "var."
        # to test if correct taxonomic id is filled in post-tnrs
        data = {'CatalogNumber': [12345, 12346, 12347, 12348, 12349],
                'fulltaxon':['x Serapicamptis', 'Castilleja miniata subsp. dixonii var. fake x fakeus',
                             'Rafflesia arnoldi var. arjehensis', 'Castilloja Moniata',
                             'Abies balsamea subsp. balsamea'],
                'taxon_id': [None, None, None, None, None],
                'Genus': ['x Serapicamptis', 'Castilleja', 'Rafflesia', 'Castilloja', 'Abies'],
                'Species': [pd.NA, 'miniata', 'arnoldi', 'Moniata', 'balsamea'],
                'Rank 1': [pd.NA, 'subsp.', 'var.', pd.NA, 'subsp.'],
                'Epithet 1': [pd.NA, 'dixonii', 'atjehensis', pd.NA, 'balsamea'],
                'Rank 2': [pd.NA, 'var.', pd.NA, pd.NA, pd.NA],
                'Epithet 2': [pd.NA, 'fake x fakeus', pd.NA, pd.NA, pd.NA],
                'Hybrid': [True, True, False, False, False],
                'missing_rank': [False, False, False, False, True]
                }

        self.test_csv_create_picturae.record_full = pd.DataFrame(data)

    def test_taxon_concat_string(self):
        """tests whether correct full taxon name string is returned from taxon_concat"""
        temp_taxon_list = []
        for index, row in self.test_csv_create_picturae.record_full.iterrows():
            self.test_csv_create_picturae.taxon_concat(row)
            temp_taxon_list.extend(self.test_csv_create_picturae.taxon_concat(row))
        self.assertEqual(temp_taxon_list[0], 'x Serapicamptis')
        self.assertEqual(temp_taxon_list[3], 'x Serapicamptis')
        self.assertEqual((temp_taxon_list[5]), 'Castilleja miniata')
        self.assertEqual((temp_taxon_list[6]), 'Castilleja miniata subsp. dixonii')
        self.assertEqual((temp_taxon_list[7]), 'Castilleja miniata subsp. dixonii')
        self.assertEqual((temp_taxon_list[8]), 'fake x fakeus')
        self.assertEqual((temp_taxon_list[9]), 'Castilleja miniata subsp. dixonii var. fake x fakeus')
        self.assertEqual((temp_taxon_list[10]), 'Rafflesia arnoldi')
        self.assertEqual((temp_taxon_list[12]), 'Rafflesia arnoldi var. atjehensis')
        self.assertEqual(len(temp_taxon_list), 25)



    def test_check_taxon_real(self):
        """tests the TNRS name resolution service in the check_taxon_real function"""

        self.test_csv_create_picturae.record_full[['gen_spec', 'fullname',
                                                   'first_intra',
                                                   'taxname', 'hybrid_base']] = \
            self.test_csv_create_picturae.record_full.apply(self.test_csv_create_picturae.taxon_concat, axis=1,
                                                            result_type='expand')

        self.test_csv_create_picturae.taxon_check_tnrs()
        # assert statements
        self.assertEqual(len(self.test_csv_create_picturae.record_full.columns), 18)

        # 3 rows left as the genus level hybrid Serapicamptis and the mispelled "Castilloja" should fail

        self.assertEqual(len(self.test_csv_create_picturae.record_full), 3)



    def test_corrected_rank_id(self):
        """tests whether post-TNRS, that a taxonomic name with a missing or incorrect rank,
            with update the taxon_id to that of the corrected taxonomic name."""

        self.test_csv_create_picturae.record_full[['gen_spec', 'fullname',
                                                   'first_intra',
                                                   'taxname', 'hybrid_base']] = \
            self.test_csv_create_picturae.record_full.apply(self.test_csv_create_picturae.taxon_concat, axis=1,
                                                            result_type='expand')

        self.test_csv_create_picturae.taxon_check_tnrs()

        # assert that correct taxonomic id was filled in for Abies balsamea var. balsamea
        self.assertEqual(self.test_csv_create_picturae.record_full.iloc[2, 2], 128210)

        self.assertTrue('name_matched' in self.test_csv_create_picturae.record_full.columns,
                        "does not contain name_matched")


    def test_iterate_taxon(self):
        """test taxonomic names with incorrect ranks"""

        test_df = {'CatalogNumber': [1, 2, 3, 4],
                   'fullname': ['Rafflesia arnoldi subsp. atjehensis', 'Bredia amoena subsp. trimera',
                                'Sparganium eurycarpum var. coreanum', 'Aechmea fosteriana var. rupicola']
                   }

        test_df = pd.DataFrame(test_df)

        resolved_taxon = iterate_taxon_resolve(test_df)

        resolved_taxon = resolved_taxon[resolved_taxon['overall_score'] >= .99]

        matched_list = list(resolved_taxon['name_matched'])

        self.assertEqual(len(matched_list), 4)

        self.assertEqual(['Rafflesia arnoldi var. atjehensis', 'Bredia amoena var. trimera',
                          'Sparganium eurycarpum subsp. coreanum', 'Aechmea fosteriana subsp. rupicola'],
                         matched_list)


    def tearDown(self):
        """deleting instance of PicturaeImporter"""
        del self.test_csv_create_picturae


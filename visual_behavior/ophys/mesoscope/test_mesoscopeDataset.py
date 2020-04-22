import unittest
from unittest import TestCase
import os
import visual_behavior.ophys.mesoscope.dataset as ms


class TestMesoscopeDataset(TestCase):

	def __init__(self, test_session=None):
		super().__init__()
		if not test_session:
			self.session = 958772311
		else:
			self.session = test_session

	def setUp(self):
		self.ds = ms.MesoscopeDataset(session_id=self.session)

	def test_get_paired_planes(self):
		pairs = self.ds.get_paired_planes()
		self.assertIsNotNone(pairs), f"Returned None instead of pairs"
		self.assertEqual(len(pairs), 4), "Returned pairs lenght is not 4"

	def test_get_splitting_json(self, test_session=None):
		"""
		test for visual_behavior.ophys.mesoscope.crosstalk_unmix.MesoscopeICA.get_ica_traces()
		Testing that:
			returned file exists
		:return:
		"""
		if not test_session:
			session = 958772311
		else:
			session = test_session

		splitting_json_path = self.ds.get_splitting_json()

		# 1. Test if a valid filepath is returned
		self.assertTrue(os.path.isfile(splitting_json_path), f"Failed find a slitting json for session {session}")


if __name__ == '__main__':
    unittest.main()

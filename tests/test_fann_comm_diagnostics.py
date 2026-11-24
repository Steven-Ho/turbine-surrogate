import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tfai.benchmarks.fann_comm_diagnostics import setup_debug,summarize_communication


class CommDiagnosticTests(unittest.TestCase):
    def test_logging_setup_preserves_network_selection(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ,{'FANN_COMM_DIAGNOSTICS_DIR':home,'RANK':'11','MCCL_IB_HCA':'existing_hca'}):
                root=setup_debug();self.assertEqual(root,Path(home))
                self.assertEqual(os.environ['MCCL_IB_HCA'],'existing_hca')
                for prefix in ('NCCL','MCCL'):
                    self.assertEqual(os.environ[prefix+'_DEBUG'],'INFO')
                    self.assertIn('rank_00011',os.environ[prefix+'_DEBUG_FILE'])
                    self.assertIn('NET',os.environ[prefix+'_DEBUG_SUBSYS'])

    def test_bootstrap_does_not_imply_socket_data_transport(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);(root/'communication').mkdir()
            (root/'communication/mccl.log').write_text('Bootstrap : Using eth0\nNET/IB : Using mlx5_0\n')
            report=json.loads(Path(summarize_communication(root)).read_text())
            self.assertEqual(len(report['bootstrap_evidence']),1)
            self.assertEqual(len(report['transport_evidence']),1)
            self.assertIn('NET/IB',report['transport_evidence'][0]['text'])

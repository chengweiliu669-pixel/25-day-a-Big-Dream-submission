import unittest
import torch
from train_temporal_state_corrector_20261102 import Corrector
class TemporalChecks(unittest.TestCase):
    def test_zero_initialization_preserves_parent_and_known_cells(self):
        torch.manual_seed(5);m=Corrector(28);x=torch.randn(3,28,24);p=torch.rand(3,2,24);mask=torch.rand(3,24)>.5
        delta=m(x);self.assertEqual(int(torch.count_nonzero(delta)),0);torch.testing.assert_close(p+delta*mask[:,None],p,rtol=0,atol=0)
        with torch.no_grad():m.output.bias.fill_(.5)
        pred=p+m(x)*mask[:,None];torch.testing.assert_close(pred.transpose(1,2)[~mask],p.transpose(1,2)[~mask],rtol=0,atol=0);self.assertTrue((m(x).abs()<=.2).all())
if __name__=='__main__':unittest.main()

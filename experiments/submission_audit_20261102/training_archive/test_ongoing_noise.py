import unittest,tempfile
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train_ongoing_noise import soft_objective
from ongoing_jaccard_core import sigmoid
class SoftTargetChecks(unittest.TestCase):
    def test_continuous_targets_weighted_gradient_and_saved_initial_score(self):
        rng=np.random.default_rng(18);x=rng.normal(size=(240,3));y=sigmoid(1.4*x[:,0]);w=rng.uniform(.1,2,len(y));z=.3*x[:,1];eps=1e-5
        data=lgb.Dataset(x,label=y,weight=w,init_score=z,free_raw_data=False,params={'feature_pre_filter':False});data.construct();label=data.get_label();g,h=soft_objective(w)(z,data)
        loss=lambda a:w*(np.logaddexp(0,a)-label*a)
        np.testing.assert_allclose(g,(loss(z+eps)-loss(z-eps))/(2*eps),atol=1e-9);assert (h>0).all()
        model=lgb.train(dict(objective=soft_objective(w),verbosity=-1,num_threads=2,num_leaves=7,min_data_in_leaf=5,learning_rate=.1),data,num_boost_round=30)
        raw=model.predict(x,raw_score=True);assert np.mean(loss(z+raw))<np.mean(loss(z))*.85
        with tempfile.TemporaryDirectory() as folder:
            f=Path(folder)/'head.txt';model.save_model(str(f));np.testing.assert_array_equal(lgb.Booster(model_file=str(f)).predict(x,raw_score=True),raw)
if __name__=='__main__':unittest.main()

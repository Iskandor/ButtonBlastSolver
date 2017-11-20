using FLAB;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sofia.Layers
{
    public class CoreLayer : BaseLayer
    {
        public CoreLayer(int p_dim, ACTIVATION p_activationFunction, TYPE p_type) : base(p_type, CORE)
        {
            NeuralGroup core = new NeuralGroup("core", p_dim, p_activationFunction);

            _groups.Add("core", core);

            _inputGroup = _outputGroup = core;
        }

        public CoreLayer(JSONObject p_data) : base(p_data)
        {

        }

        override public Vector Activate(Vector p_input = null)
        {
            ActivateGroup("core", true, false);            

            return _outputGroup.Output;
        }

        override public Tensor ActivateT(Tensor p_input = null)
        {
            foreach(Connection c in _inputGroup.GetInConnections())
            {
                /* osetrit potom pre viac vstupov */
                _groups["core"].Batch = (int)c.InGroup.OutputT.GetShape(0);
            }
            
            ActivateGroup("core", true, true);

            return _outputGroup.OutputT;
        }

        override public Dictionary<string, Matrix> CalcGradient(Vector p_error = null)
        {
            NeuralGroup core = _groups["core"];

            core.CalcDerivs(false);
            if (p_error == null)
            {
                core.CalcDelta(false);
            }
            else
            {
                Vector e = Vector.HadamardProduct(core.Derivs, p_error);
                Vector.Release(core.Delta);
                core.Delta = e;
            }

            foreach (Connection c in core.GetInConnections())
            {
                if (c.Trainable)
                {
                    c.CalcGradient(false);
                    _gradients[c.Id] = c.Gradient;
                }
            }

            return _gradients;
        }

        override public Dictionary<string, Tensor> CalcGradientT(Tensor p_error = null)
        {
            NeuralGroup core = _groups["core"];

            core.CalcDerivs(true);
           
            if (p_error == null)
            {
                core.CalcDelta(true);
            }
            else
            {
                core.DeltaT = core.DerivsT * p_error;
            }

            foreach (Connection c in core.GetInConnections())
            {
                if (c.Trainable)
                {
                    c.CalcGradient(true);
                    _gradients_t[c.Id] = c.GradientT;
                }
            }

            return _gradients_t;
        }

        public override JSONObject GetFileData()
        {
            JSONObject data = base.GetFileData();

            data.AddField("type", CORE);
            data.AddField("connections", GetConnectionFileData());

            return data;
        }
    }
}

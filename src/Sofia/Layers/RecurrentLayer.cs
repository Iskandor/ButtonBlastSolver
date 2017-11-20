using FLAB;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sofia.Layers
{
    public class RecurrentLayer : BaseLayer
    {
        public RecurrentLayer(int p_dim, ACTIVATION p_activationFunction, TYPE p_type) : base(p_type, RECURRENT)
        {
            NeuralGroup core = new NeuralGroup("core", p_dim, p_activationFunction);
            NeuralGroup context = new NeuralGroup("context", p_dim, ACTIVATION.IDENTITY);

            _groups.Add("core", core);
            _groups.Add("context", context);

            Connection c = Connect(context, core, true);
            c.Init(Connection.INIT.LECUN_UNIFORM, 0.05f);

            Connection r = Connect(core, context, false);
            r.Weights = Matrix.Identity(p_dim, p_dim);

            _inputGroup = _outputGroup = core;
        }

        public RecurrentLayer(JSONObject p_data) : base(p_data)
        {

        }

        override public Vector Activate(Vector p_input = null)
        {
            ActivateGroup("context", false, false);
            ActivateGroup("core", true, false);

            return _outputGroup.Output;
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
                core.Delta = Vector.HadamardProduct(core.Derivs, p_error);
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

        public override void ResetContext()
        {
            _groups["core"].Output.Fill(0);
        }

        public override JSONObject GetFileData()
        {
            JSONObject data = base.GetFileData();

            data.AddField("type", RECURRENT);
            data.AddField("connections", GetConnectionFileData());

            return data;
        }
    }
}

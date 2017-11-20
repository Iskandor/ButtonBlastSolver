using FLAB;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sofia.Layers
{
    public class InputLayer : BaseLayer
    {
        public InputLayer(int p_dim) : base(TYPE.INPUT, INPUT)
        {            
            NeuralGroup core = new NeuralGroup("input", p_dim, ACTIVATION.IDENTITY);

            _groups.Add("input", core);

            _inputGroup = _outputGroup = core;
        }

        public InputLayer(JSONObject p_data) : base(p_data)
        {
            NeuralGroup input = _groups["input"];
        }

        override public Vector Activate(Vector p_input)
        {
            NeuralGroup core = _groups["input"];

            if (p_input != null)
            {
                core.Output = p_input;
            }

            return core.Output;
        }

        override public Tensor ActivateT(Tensor p_input)
        {
            NeuralGroup core = _groups["input"];

            if (p_input != null)
            {
                core.Batch = (int)p_input.GetShape(0);
                core.OutputT = p_input;
            }

            return core.OutputT;
        }

        public override JSONObject GetFileData()
        {
            JSONObject data = base.GetFileData();

            data.AddField("type", INPUT);

            return data;
        }
    }
}

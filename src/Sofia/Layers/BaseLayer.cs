using FLAB;
using System;
using System.Collections.Generic; 

namespace Sofia.Layers
{
    public class BaseLayer
    {
        public enum TYPE
        {
            HIDDEN = 0,
            INPUT = 1,
            OUTPUT = 2,
            CONTEXT = 3
        };

        public const string INPUT = "input";
        public const string CORE = "core";
        public const string RECURRENT = "recurrent";
        public const string LSTM = "lstm";

        private string _id;
        private TYPE _type;
        private string _architecture;
        private bool _valid;

        protected NeuralGroup _inputGroup;
        protected NeuralGroup _outputGroup;

        protected Dictionary<string, NeuralGroup> _groups;
        protected Dictionary<string, Connection> _connections;

        protected Dictionary<string, Matrix> _gradients;
        protected Dictionary<string, Tensor> _gradients_t;

        public BaseLayer(TYPE p_type, string p_architecture)
        {
            _architecture = p_architecture;
            _groups = new Dictionary<string, NeuralGroup>();
            _connections = new Dictionary<string, Connection>();
            _gradients = new Dictionary<string, Matrix>();
            _gradients_t = new Dictionary<string, Tensor>();
            _type = p_type;
        }

        public BaseLayer(JSONObject p_data)
        {
            _gradients = new Dictionary<string, Matrix>();

            JSONObject groups = p_data["groups"];

            _id = p_data["id"].str;
            _type = (TYPE)Enum.Parse(typeof(TYPE), p_data["layer_type"].str);

            _groups = new Dictionary<string, NeuralGroup>();

            foreach (string key in groups.keys)
            {
                JSONObject layer = groups[key];

                NeuralGroup g = new NeuralGroup(key, layer);

                _groups.Add(key, g);

                if (key.Equals(p_data["ingroup"].str))
                {
                    _inputGroup = g;
                }
                if (key.Equals(p_data["outgroup"].str))
                {
                    _outputGroup = g;
                }
            }

            _connections = new Dictionary<string, Connection>();

            if (p_data.HasField("connections"))
            {
                JSONObject connections = p_data["connections"];

                foreach (string key in connections.keys)
                {
                    JSONObject connection = connections[key];

                    NeuralGroup inGroup = connection.HasField("ingroup") ? _groups[connection["ingroup"].str] : null;
                    NeuralGroup outGroup = _groups[connection["outgroup"].str];

                    Connection c = new Connection(key, inGroup, outGroup, connection);

                    _connections.Add(key, c);

                    if (inGroup != null) inGroup.AddOutConnection(c);
                    outGroup.AddInConnection(c);
                }
            }
        }

        public void Dispose()
        {
            foreach (NeuralGroup g in _groups.Values)
            {
                g.Dispose();
            }
            foreach (Connection c in _connections.Values)
            {
                c.Dispose();
            }
        }

        public void ReinitParams()
        {
            foreach (NeuralGroup g in _groups.Values)
            {
                Vector.Release(g.Bias);
                g.Bias = Vector.Random(g.Dim);
            }
        }

        public void OverrideParams(BaseLayer p_source)
        {
            foreach (NeuralGroup g in p_source._groups.Values)
            {
                Vector.Release(_groups[g.Id].Bias);
                _groups[g.Id].Bias = Vector.Copy(g.Bias);
            }
        }

        protected Connection Connect(NeuralGroup p_g1, NeuralGroup p_g2, bool p_trainable = true) {
            Connection c = new Connection(p_g1, p_g2, p_trainable);
            p_g1.AddOutConnection(c);
            p_g2.AddInConnection(c);
            _connections.Add(c.Id, c);

            return c;
        }

        protected void ActivateGroup(string p_group, bool p_bias, bool p_tensor)
        {
            NeuralGroup group = _groups[p_group];
            foreach (Connection c in group.GetInConnections())
            {
                if (p_tensor)
                {                    
                    group.Integrate(c.InGroup.OutputT, c.Weights);
                }
                else
                {
                    group.Integrate(c.InGroup.Output, c.Weights);
                }                
            }
            if (p_bias) group.AddBias(p_tensor);
            group.Activate(p_tensor);
        }

        public NeuralGroup InputGroup
        {
            get { return _inputGroup; }
        }

        public NeuralGroup OutputGroup
        {
            get { return _outputGroup; }
        }

        virtual public Vector Activate(Vector p_input = null)
        {
            return null;
        }

        virtual public Tensor ActivateT(Tensor p_input = null)
        {
            return null;
        }

        virtual public Dictionary<string, Matrix> CalcGradient(Vector p_error = null)
        {
            return null;
        }

        virtual public Dictionary<string, Tensor> CalcGradientT(Tensor p_error = null)
        {
            return null;
        }

        virtual public void ResetContext()
        {
        }

        public string Id
        {
            get { return _id; }
            set { _id = value; }
        }

        public bool Valid
        {
            get { return _valid; }
            set { _valid = value; }
        }

        public TYPE Type
        {
            get { return _type; }
            set { _type = value; }
        }

        public string Architecture
        {
            get { return _architecture; }
        }

        public Dictionary<string, Connection> Connections
        {
            get { return _connections; }
        }

        virtual public JSONObject GetFileData()
        {
            Dictionary<string, string> data = new Dictionary<string, string> { {"id", _id }, {"layer_type", _type.ToString() }, { "ingroup", _inputGroup.Id }, { "outgroup", _outputGroup.Id } };

            JSONObject result = new JSONObject(data);

            Dictionary<string, JSONObject> groups = new Dictionary<string, JSONObject>();

            foreach (NeuralGroup g in _groups.Values)
            {
                groups[g.Id] = g.GetFileData();
            }

            result.AddField("groups", new JSONObject(groups));

            return result;
        }

        public JSONObject GetConnectionFileData()
        {
            Dictionary<string, JSONObject> connections = new Dictionary<string, JSONObject>();

            foreach (Connection c in _connections.Values)
            {
                connections[c.Id] = c.GetFileData();
            }

            return new JSONObject(connections);
        }
    }
}

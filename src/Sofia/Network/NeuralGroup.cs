using FLAB;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace Sofia
{
    public enum ACTIVATION
    {
        IDENTITY = 0,
        BIAS = 1,
        BINARY = 2,
        SIGMOID = 3,
        TANH = 4,
        SOFTMAX = 5,
        LINEAR = 6,
        EXPONENTIAL = 7,
        SOFTPLUS = 8,
        RELU = 9
    };

    public class NeuralGroup
    {
        private string _id;
        private int _dim;
        private int _batch;
        private ACTIVATION _activationFunction;
        private bool _valid;
        private bool _dropoutSwitch;

        private Vector _output;
        private Vector _derivs;
        private Vector _ap;
        private Vector _bias;
        private Vector _delta;

        private Tensor _output_t;
        private Tensor _derivs_t;
        private Tensor _ap_t;
        private Tensor _bias_t;
        private Tensor _delta_t;

        private List<Connection> _inConnections;
        private List<Connection> _outConnection;

        public NeuralGroup(string p_id, int p_dim, ACTIVATION p_activationFunction)
        {
            _id = p_id;
            _dim = p_dim;
            _batch = 0;
            _activationFunction = p_activationFunction;
            _inConnections = new List<Connection>();
            _outConnection = new List<Connection>();

            _output = Vector.Zero(_dim);
            _output_t = null;
            _ap = Vector.Zero(_dim);
            _ap_t = null;
            _bias = Vector.Random(_dim);
            _bias_t = null;
            //_bias = Vector.Zero(_dim);
            _derivs = null;
            _derivs_t = null;
            _valid = false;
        }

        public NeuralGroup(string p_id, JSONObject p_data)
        {
            _id = p_id;
            _dim = int.Parse(p_data["dim"].str);
            _activationFunction = (ACTIVATION)Enum.Parse(typeof(ACTIVATION), p_data["actfn"].str);

            _inConnections = new List<Connection>();
            _outConnection = new List<Connection>();

            _output = Vector.Zero(_dim);
            _ap = Vector.Zero(_dim);
            _bias = BasePool.Instance.Get(_dim);

            byte[] buffer = Convert.FromBase64String(p_data["bias"].str);

            MemoryStream stream = new MemoryStream(buffer);
            BinaryReader reader = new BinaryReader(stream);

            for (int i = 0; i < _dim; i++)
            {
                _bias[i] = reader.ReadSingle();
            }

            _derivs = null;
            _valid = false;
        }

        public void Dispose()
        {
            Vector.Release(_output);
            Vector.Release(_ap);
            Vector.Release(_bias);

            Vector.Release(_derivs);
            Vector.Release(_delta);
        }

        public void AddInConnection(Connection p_connection)
        {
            _inConnections.Add(p_connection);
        }

        public void AddOutConnection(Connection p_connection)
        {
            _outConnection.Add(p_connection);
        }

        public void Integrate(Vector p_input, Matrix p_weights)
        {
            Vector dap = p_weights * p_input;
            Vector ap = _ap + dap;
            Vector.Release(_ap);
            Vector.Release(dap);
            _ap = ap;
        }

        public void Integrate(Tensor p_input, Matrix p_weights)
        {
            if (_ap_t == null)
            {
                _ap_t = (new Tensor(p_weights, _batch) * p_input);               
            }
            else
            {
                _ap_t += (new Tensor(p_weights, _batch) * p_input);
            }
        }

        public void AddBias(bool p_tensor)
        {
            if (p_tensor)
            {
                _bias_t = new Tensor(_bias, _batch);
                _ap_t += _bias_t;
            }
            else
            {
                Vector ap = _ap + _bias;
                Vector.Release(_ap);
                _ap = ap;                
            }
        }

        public void UpdateBias(Vector p_update)
        {
            Vector db = _bias + p_update;
            Vector.Release(_bias);
            _bias = db;
        }

        public void UpdateBias(Tensor p_bias)
        {
            _bias += Vector.Build(_bias.Size, p_bias.ReduceSum(0).Buffer);
        }

        public void Activate(bool p_tensor)
        {
            if (p_tensor)
            {
                _output_t = new Tensor(_ap_t);
                switch (_activationFunction)
                {
                    case ACTIVATION.IDENTITY:
                    case ACTIVATION.LINEAR:
                        _output_t.Apply(Activation.Linear);
                        break;
                    case ACTIVATION.BINARY:
                        _output_t.Apply(Activation.Binary);
                        break;
                    case ACTIVATION.SIGMOID:
                        _output_t.Apply(Activation.Sigmoid);
                        break;
                    case ACTIVATION.TANH:
                        _output_t.Apply(Activation.Tanh);
                        break;
                    case ACTIVATION.SOFTMAX:
                        /* TODO SOFTMAX premysliet */
                        break;
                    case ACTIVATION.SOFTPLUS:
                        _output_t.Apply(Activation.Softplus);
                        break;
                    case ACTIVATION.RELU:
                        _output_t.Apply(Activation.ReLU);
                        break;
                }
                _ap_t.Fill(0);
            }
            else
            {
                Vector.Release(_output);
                switch (_activationFunction)
                {
                    case ACTIVATION.IDENTITY:
                    case ACTIVATION.LINEAR:
                        _output = Vector.Apply(_ap, Activation.Linear);
                        break;
                    case ACTIVATION.BINARY:
                        _output = Vector.Apply(_ap, Activation.Binary);
                        break;
                    case ACTIVATION.SIGMOID:
                        _output = Vector.Apply(_ap, Activation.Sigmoid);
                        break;
                    case ACTIVATION.TANH:
                        _output = Vector.Apply(_ap, Activation.Tanh);
                        break;
                    case ACTIVATION.SOFTMAX:
                        /*
                        {
                            double sumExp = 0;
                            for (int i = 0; i < _dim; i++)
                            {
                                sumExp += Math.Exp(_ap[i]);
                            }
                            _output[index] = (float)(Math.Exp(_ap[index]) / sumExp);
                            _ap[index] = 0;
                        }
                        */
                        break;
                    case ACTIVATION.SOFTPLUS:
                        _output = Vector.Apply(_ap, Activation.Softplus);
                        break;
                    case ACTIVATION.RELU:
                        _output = Vector.Apply(_ap, Activation.ReLU);
                        break;
                }
                _ap.Fill(0f);
            }
        }

        public void CalcDerivs(bool p_tensor)
        {
            if (p_tensor)
            {
                Tensor deriv = new Tensor(_output_t);

                switch (_activationFunction)
                {
                    case ACTIVATION.IDENTITY:
                    case ACTIVATION.BINARY:
                    case ACTIVATION.LINEAR:
                        _derivs_t = new Tensor(Matrix.Identity(_dim, _dim), _batch);
                        break;
                    case ACTIVATION.SIGMOID:
                        deriv.Apply(Activation.dSigmoid);
                        _derivs_t = Tensor.Diag(deriv);
                        break;
                    case ACTIVATION.TANH:
                        deriv.Apply(Activation.dTanh);
                        _derivs_t = Tensor.Diag(deriv);
                        break;
                    case ACTIVATION.SOFTMAX:
                        /* TODO SOFTMAX premysliet */
                        break;
                    case ACTIVATION.SOFTPLUS:
                        deriv.Apply(Activation.dSoftplus);
                        _derivs_t = Tensor.Diag(deriv);
                        break;
                    case ACTIVATION.RELU:
                        deriv.Apply(Activation.dReLU);
                        _derivs_t = Tensor.Diag(deriv);
                        break;
                }
            }
            else
            {
                Vector.Release(_derivs);
                switch (_activationFunction)
                {
                    case ACTIVATION.IDENTITY:
                    case ACTIVATION.BINARY:
                    case ACTIVATION.LINEAR:
                        _derivs = Vector.One(_dim);
                        break;
                    case ACTIVATION.SIGMOID:
                        _derivs = Vector.Apply(_output, Activation.dSigmoid);
                        break;
                    case ACTIVATION.TANH:
                        _derivs = Vector.Apply(_output, Activation.dTanh);
                        break;
                    case ACTIVATION.SOFTMAX:
                        /*
                        for (int i = 0; i < _dim; i++)
                        {
                            for (int j = 0; j < _dim; j++)
                            {
                                _derivs[i, j] = _output[i] * (NetworkUtils.KroneckerDelta(i, j) - _output[j]);
                            }
                        }
                        */
                        break;
                    case ACTIVATION.SOFTPLUS:
                        _derivs = Vector.Apply(_output, Activation.dSoftplus);
                        break;
                    case ACTIVATION.RELU:
                        _derivs = Vector.Apply(_output, Activation.dReLU);
                        break;
                }
            }
        }

        public void CalcDelta(bool p_tensor)
        {
            if (p_tensor)
            {
                _delta_t = new Tensor(_batch, _dim);

                foreach (Connection c in _outConnection)
                {
                    if (c.Trainable)
                    {
                        Tensor wT = new Tensor(c.Weights.T(), _batch);
                        _delta_t += _derivs_t * (wT * c.OutGroup._delta_t);
                    }
                }
            }
            else
            {
                Vector.Release(_delta);
                _delta = Vector.Zero(_dim);

                foreach (Connection c in _outConnection)
                {
                    if (c.Trainable)
                    {
                        Matrix wT = c.Weights.T();
                        Vector d1 = wT * c.OutGroup._delta;
                        Vector d2 = Vector.HadamardProduct(_derivs, d1);
                        Vector d3 = _delta + d2;
                        Matrix.Release(wT);
                        Vector.Release(d1);
                        Vector.Release(d2);
                        Vector.Release(_delta);
                        _delta = d3;                        
                    }
                }
            }
        }

        public JSONObject GetFileData()
        {
            Dictionary<string, string> data = new Dictionary<string, string>();
            string bias = string.Empty;

            MemoryStream stream = new MemoryStream();
            BinaryWriter writer = new BinaryWriter(stream);

            long d = DateTime.Now.ToFileTime();

            for (int i = 0; i < _dim; i++)
            {
                writer.Write(_bias[i]);

            }

            bias = Convert.ToBase64String(stream.ToArray());

            data["dim"] = _dim.ToString();
            data["actfn"] = _activationFunction.ToString();
            data["bias"] = bias;

            return new JSONObject(data);
        }

        public string Id
        {
            get { return _id; }
        }

        public Vector Output
        {
            get { return _output; }
            set { _output = value; } 
        }

        public Tensor OutputT
        {
            get { return _output_t; }
            set { _output_t = value; }
        }

        public Vector Derivs
        {
            get { return _derivs; }
        }

        public Tensor DerivsT
        {
            get { return _derivs_t; }
        }

        public Vector Bias
        {
            get { return _bias; }
            set { _bias = value; }
        }

        public Tensor BiasT
        {
            get { return _bias_t; }
            set { _bias_t = value; }
        }

        public Vector Delta
        {
            get { return _delta; }
            set { _delta = value; }
        }

        public Tensor DeltaT
        {
            get { return _delta_t; }
            set { _delta_t = value; }
        }

        public List<Connection> GetOutConnections()
        {
            return _outConnection;
        }

        public List<Connection> GetInConnections()
        {
            return _inConnections;
        }

        public bool Valid
        {
            get { return _valid; }
            set { _valid = value; }
        }

        public int Dim
        {
            get { return _dim; }
        }

        public int Batch
        {
            set { _batch = value; }
            get { return _batch; }
        }

        public void SetDropout(bool p_val)
        {
            _dropoutSwitch = p_val;
        }
    }
}

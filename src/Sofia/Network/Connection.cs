using FLAB;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace Sofia
{
    public class Connection
    {
        public enum INIT
        {
            UNIFORM = 0,
            LECUN_UNIFORM = 1,
            GLOROT_UNIFORM = 2,
            IDENTITY = 3
        }

        private string _id;
        private NeuralGroup _inGroup;
        private NeuralGroup _outGroup;
        private string _inLayer;
        private string _outLayer;
        private int _inDim, _outDim;
        private Matrix _weights;
        private Matrix _gradient;
        private Tensor _gradient_t;
        private bool _trainable;

        public Connection(NeuralGroup p_inGroup, NeuralGroup p_outGroup, bool p_trainable, string p_inLayer = null, string p_outLayer = null)
        {
            _id = GenID();
            _inGroup = p_inGroup;
            _outGroup = p_outGroup;
            if (p_inGroup != null) {
                _inDim = p_inGroup.Dim;
            }
            else {
                _inDim = p_outGroup.Dim;
            }
            _outDim = p_outGroup.Dim;
            _weights = BasePool.Instance.Get(_outDim, _inDim);
            _trainable = p_trainable;

            _inLayer = p_inLayer;
            _outLayer = p_outLayer;
        }

        public Connection(string p_id, NeuralGroup p_inGroup, NeuralGroup p_outGroup, JSONObject p_data)
        {
            _id = p_id;
            _inGroup = p_inGroup;
            _outGroup = p_outGroup;
            if (p_inGroup != null)
            {
                _inDim = p_inGroup.Dim;
            }
            else
            {
                _inDim = p_outGroup.Dim;
            }
            _outDim = p_outGroup.Dim;

            int nRows = _outGroup.Dim;
            int nCols = _inGroup == null ? _outGroup.Dim : _inGroup.Dim; 

            _weights = BasePool.Instance.Get(nRows, nCols);

            byte[] buffer = Convert.FromBase64String(p_data["weights"].str);

            MemoryStream stream = new MemoryStream(buffer);
            BinaryReader reader = new BinaryReader(stream);

            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    _weights[i, j] = reader.ReadSingle();
                }
            }

            _trainable = p_data["trainable"].str.Equals("1");

            _inLayer = p_data.HasField("inlayer") ? p_data["inlayer"].str : null;
            _outLayer = p_data.HasField("outlayer") ? p_data["outlayer"].str : null;
        }

        public void Dispose()
        {
            Matrix.Release(_weights);
            Matrix.Release(_gradient);
        }

        public void CalcGradient(bool p_tensor)
        {
            if (_outGroup.DeltaT != null)
            {
                _gradient_t = _outGroup.DeltaT * _inGroup.OutputT;
            }
            else
            {
                Vector ot = _inGroup.Output.T(); // T kvoli vector outer product
                Matrix g = _outGroup.Delta * ot;
                Vector.Release(ot);
                Matrix.Release(_gradient);
                _gradient = g; 
            }            
        }

        public void Init(INIT p_init, float p_limit)
        {
            switch (p_init)
            {
                case INIT.UNIFORM:
                    Uniform(p_limit);
                    break;
                case INIT.LECUN_UNIFORM:
                    Uniform((float)Math.Pow(_inDim, -.5));
                    break;
                case INIT.GLOROT_UNIFORM:
                    Uniform(2f / (_inDim + _outDim));
                    break;
                case INIT.IDENTITY:
                    Identity();
                    break;
            }
        }

        public void Update(Tensor p_update)
        {
            _weights += Matrix.Build(_weights.Rows, _weights.Cols, p_update.ReduceSum(0).Buffer);
        }

        public void Update(Matrix p_update)
        {
            Matrix dw = _weights + p_update;
            Matrix.Release(_weights);
            _weights = dw;
        }

        public JSONObject GetFileData()
        {
            Dictionary<string, string> data = new Dictionary<string, string>();
            string weights = string.Empty;  //string.Join("|", _weights);

            MemoryStream stream = new MemoryStream();
            BinaryWriter writer = new BinaryWriter(stream);

            long d = DateTime.Now.ToFileTime();

            for (int i = 0; i < _outDim; i++)
            {
                for (int j = 0; j < _inDim; j++)
                {
                    writer.Write(_weights[i, j]);
                }
            }

            weights = Convert.ToBase64String(stream.ToArray());

            data["trainable"] = _trainable ? "1" : "0";
            if (_inGroup != null) data["ingroup"] = _inGroup.Id;
            data["outgroup"] = _outGroup.Id;
            data["weights"] = weights;
            if (_inLayer != null) data["inlayer"] = _inLayer;
            if (_outLayer != null) data["outlayer"] = _outLayer;

            return new JSONObject(data);
        }

        public byte[] GetBinData()
        {
            MemoryStream stream = new MemoryStream();
            BinaryWriter writer = new BinaryWriter(stream);

            writer.Write(_trainable);
            //writer.Write(_inGroup.Id);
            writer.Write(_outGroup.Id);

            for (int i = 0; i < _outDim; i++)
            {
                for (int j = 0; j < _inDim; j++)
                {
                    writer.Write(_weights[i, j]);
                }
            }

            //writer.Write(_inLayer);
            //writer.Write(_outLayer);

            return stream.GetBuffer();
        }

        private void Uniform(float p_limit)
        {
            for (int i = 0; i < _outDim; i++)
            {
                for (int j = 0; j < _inDim; j++)
                {
                    _weights[i,j] = RandomGenerator.getInstance().Rand(-p_limit, p_limit);
                    //_weights[i, j] = 0;
                }
            }
        }

        private string GenID()
        {
            return Guid.NewGuid().ToString("N");
        }

        private void Identity()
        {
            _weights = Matrix.Identity(_outDim, _inDim);
        }

        public string Id
        {
            get { return _id; }
        }

        public bool Trainable 
        {
            get { return _trainable; }
        }

        public Matrix Weights
        {
            get { return _weights; }
            set { _weights = value; }
        }

        public Matrix Gradient
        {
            get { return _gradient; }
        }

        public Tensor GradientT
        {
            get { return _gradient_t; }
        }

        public NeuralGroup OutGroup
        {
            get { return _outGroup; }
        }

        public NeuralGroup InGroup
        {
            get { return _inGroup; }
        }

        public int OutDim
        {
            get { return _outDim; }
        }

        public int InDim
        {
            get { return _inDim; }
        }

        public string InLayer
        {
            get { return _inLayer; }
        }

        public string OutLayer
        {
            get { return _outLayer; }
        }
    }
}

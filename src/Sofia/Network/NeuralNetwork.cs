using FLAB;
using Sofia.Layers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Sofia
{
    public class NeuralNetwork
    {
        protected Dictionary<string, Connection> _connections;
        protected Dictionary<string, BaseLayer> _layers;
        protected Dictionary<string, Matrix> _gradient;
        protected Dictionary<string, Tensor> _gradient_t;

        BaseLayer _inputLayer;
        BaseLayer _outputLayer;
        List<BaseLayer> _forwardGraph;
        List<BaseLayer> _backwardGraph;

        public NeuralNetwork()
        {
            _layers = new Dictionary<string, BaseLayer>();
            _connections = new Dictionary<string, Connection>();
            _gradient = new Dictionary<string, Matrix>();
            _gradient_t = new Dictionary<string, Tensor>();
        }

        public void Dispose()
        {
            foreach (BaseLayer l in _layers.Values)
            {
                l.Dispose();
            }

            foreach (Connection c in _connections.Values)
            {
                c.Dispose();
            }
        }

        private void CreateGraphBase()
        {
            List<BaseLayer> graph = new List<BaseLayer>();
            foreach (BaseLayer l in _layers.Values)
            {
                l.Valid = false;
            }

            Queue<BaseLayer> q = new Queue<BaseLayer>();

            q.Enqueue(_outputLayer);

            while (q.Count > 0)
            {
                BaseLayer v = q.Dequeue();

                if (!v.Valid)
                {
                    graph.Add(v);
                    v.Valid = true;

                    foreach(BaseLayer l in CreateLayerList(v.InputGroup.GetInConnections()))
                    {
                        if (!l.Valid)
                        {
                            q.Enqueue(l);
                        }                        
                    }
                }
            }


            _backwardGraph = new List<BaseLayer>(graph);
            graph.Reverse();
            _forwardGraph = new List<BaseLayer>(graph);
        }

        private List<BaseLayer> CreateLayerList(List<Connection> p_connections)
        {
            List<BaseLayer> result = new List<BaseLayer>();

            foreach(Connection c in p_connections)
            {
                foreach(BaseLayer l in _layers.Values)
                {
                    if (c.InGroup == l.OutputGroup)
                    {
                        result.Add(l);
                    }
                }
            }

            return result;
        }

        public Vector Activate(Vector p_input)
        {
            _inputLayer.Activate(p_input);

            foreach (BaseLayer l in _forwardGraph)
            {
                l.Activate();
            }

            return _outputLayer.OutputGroup.Output;
        }

        public Tensor Activate(Tensor p_input)
        {
            _inputLayer.ActivateT(p_input);

            foreach (BaseLayer l in _forwardGraph)
            {
                l.ActivateT();
            }

            return _outputLayer.OutputGroup.OutputT;
        }

        public Dictionary<string, Matrix> CalcGradient(Vector p_error)
        {
            _gradient = new Dictionary<string, Matrix>();
            _gradient = ConcatDictionary(_gradient, _outputLayer.CalcGradient(p_error));

            foreach (BaseLayer l in _backwardGraph)
            {
                if (l != _outputLayer)
                {
                    _gradient = ConcatDictionary(_gradient, l.CalcGradient());
                }                
            }

            return _gradient;
        }

        public Dictionary<string, Tensor> CalcGradient(Tensor p_error)
        {
            _gradient_t = new Dictionary<string, Tensor>();
            _gradient_t = ConcatDictionary(_gradient_t, _outputLayer.CalcGradientT(p_error));

            foreach (BaseLayer l in _backwardGraph)
            {
                if (l != _outputLayer)
                {
                    _gradient_t = ConcatDictionary(_gradient_t, l.CalcGradientT());
                }
            }

            return _gradient_t;
        }


        public void AddLayer(string p_id, BaseLayer p_layer, BaseLayer.TYPE p_type)
        {
            p_layer.Id = p_id;
            _layers.Add(p_id, p_layer);

            if (p_type == BaseLayer.TYPE.INPUT)
            {
                _inputLayer = p_layer;
            }
            if (p_type == BaseLayer.TYPE.OUTPUT)
            {
                _outputLayer = p_layer;
            }

            _connections = _connections.Concat(p_layer.Connections).ToDictionary(x => x.Key, x => x.Value);
        }

        public Connection AddConnection(string p_inLayerId, string p_outLayerId, Connection.INIT p_init = Connection.INIT.UNIFORM, bool p_trainable = true, float p_limit = 0.05f)
        {
            NeuralGroup inGroup = _layers[p_inLayerId].OutputGroup;
            NeuralGroup outGroup = _layers[p_outLayerId].InputGroup;

            Connection connection = new Connection(inGroup, outGroup, p_trainable, _layers[p_inLayerId].Id, _layers[p_outLayerId].Id);

            connection.Init(p_init, p_limit);
            _connections.Add(connection.Id, connection);
            if (inGroup != null) inGroup.AddOutConnection(connection);
            if (outGroup != null) outGroup.AddInConnection(connection);

            CreateGraphBase();

            return connection;
        }

        public void AddConnection(string p_inLayerId, string p_outLayerId, Connection p_connecion)
        {
            NeuralGroup inGroup = _layers[p_inLayerId].OutputGroup;
            NeuralGroup outGroup = _layers[p_outLayerId].InputGroup;

            _connections.Add(p_connecion.Id, p_connecion);
            if (inGroup != null) inGroup.AddOutConnection(p_connecion);
            if (outGroup != null) outGroup.AddInConnection(p_connecion);

            CreateGraphBase();
        }

        public void OverrideParams(NeuralNetwork p_source)
        {
            foreach (Connection c in p_source.Connections.Values)
            {
                Matrix.Release(GetConnection(c.InLayer, c.OutLayer).Weights);
                GetConnection(c.InLayer, c.OutLayer).Weights = Matrix.Copy(c.Weights);
            }

            foreach (BaseLayer l in p_source.Layers.Values)
            {
                _layers[l.Id].OverrideParams(l);
            }
        }

        /*
        public void MergeParams(NeuralNetwork p_source)
        {
            foreach (string gid in p_source.Groups.Keys)
            {
                _groups[gid].Bias += p_source.Groups[gid].Bias;
                _groups[gid].Bias = _groups[gid].Bias * 0.5;
            }

            foreach (int cid in p_source.Connections.Keys)
            {
                _connections[cid].Weights += p_source.Connections[cid].Weights;
                _connections[cid].Weights = _connections[cid].Weights * 0.5;
            }
        }
        */

        public Connection GetConnection(string p_inLayerId, string p_outLayerId)
        {
            Connection result = null;
            foreach (Connection c in _connections.Values)
            {
                if (c.InLayer == p_inLayerId && c.OutLayer == p_outLayerId)
                {
                    result = c;
                }
            }
            return result;
        }

        private Dictionary<string, T> ConcatDictionary<T>(Dictionary<string, T> d1, Dictionary<string, T> d2)
        {
            Dictionary<string, T> result = new Dictionary<string, T>();


            if (d1 != null)
            {
                foreach (string k in d1.Keys)
                {
                    result.Add(k, d1[k]);
                }
            }

            if (d2 != null)
            {
                foreach (string k in d2.Keys)
                {
                    result.Add(k, d2[k]);
                }
            }

            return result;
        }

        public void ReinitParams(Connection.INIT p_init, float p_limit)
        {
            foreach (Connection c in _connections.Values)
            {
                c.Init(p_init, p_limit);
            }

            foreach (BaseLayer l in _layers.Values)
            {
                _layers[l.Id].ReinitParams();
            }
        }

        public void ResetContext()
        {
            foreach(BaseLayer l in _layers.Values)
            {
                l.ResetContext();
            }
        }

        public Vector Output
        {
            get { return _outputLayer.OutputGroup.Output; }
        }

        public Tensor OutputT
        {
            get { return _outputLayer.OutputGroup.OutputT; }
        }

        public Dictionary<string, BaseLayer> Layers
        {
            get { return _layers; }
        }

        public Dictionary<string, Connection> Connections
        {
            get { return _connections; }
        }
        
        public JSONObject GetFileData()
        {
            Dictionary<string, string> data = new Dictionary<string, string>{ {"type", "feedforward" }, { "inlayer", _inputLayer.Id}, { "outlayer", _outputLayer.Id } };
            return new JSONObject(data);
        }

        public byte[] GetBinData()
        {
            MemoryStream stream = new MemoryStream();
            BinaryWriter writer = new BinaryWriter(stream);

            writer.Write(@"feedforward");
            writer.Write(_inputLayer.Id);
            writer.Write(_outputLayer.Id);

            return stream.GetBuffer();
        }
    }
}

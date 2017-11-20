using FLAB;
using Sofia.Layers;
using System;
using System.Collections.Generic;

namespace Sofia
{
    public class IOUtils
    {
        public static string SaveNetwork(NeuralNetwork p_network)
        {
            JSONObject data = new JSONObject();

            data.AddField("_header", "NeuroNet");
            data.AddField("_network", p_network.GetFileData());

            Dictionary<string, JSONObject> layers = new Dictionary<string, JSONObject>();

            foreach (string lid in p_network.Layers.Keys)
            {                
                layers[lid] = p_network.Layers[lid].GetFileData();
            }

            data.AddField("layers", new JSONObject(layers));

            Dictionary<string, JSONObject> connections = new Dictionary<string, JSONObject>();

            foreach (Connection connection in p_network.Connections.Values)
            {
                if (connection.InLayer != null && connection.OutLayer != null)
                {
                    connections[connection.Id.ToString()] = connection.GetFileData();
                }                
            }

            data.AddField("connections", new JSONObject(connections));

            return data.ToString();
        }

        public static NeuralNetwork LoadNetwork(string p_data)
        {
            NeuralNetwork res = null;

            string data = p_data;

            JSONObject main = new JSONObject(data);
            JSONObject network = main["_network"];

            if (network["type"].str.Equals("feedforward"))
            {
                res = new NeuralNetwork();

                string inGroupId = network["inlayer"].str;
                string outGroupId = network["outlayer"].str;

                JSONObject layers = main["layers"];

                foreach (string key in layers.keys)
                {
                    JSONObject layer = layers[key];

                    switch(layer["type"].str)
                    {
                        case BaseLayer.INPUT:
                            res.AddLayer(layer["id"].str, new InputLayer(layer), (BaseLayer.TYPE)Enum.Parse(typeof(BaseLayer.TYPE), layer["layer_type"].str));
                            break;
                        case BaseLayer.CORE:
                            res.AddLayer(layer["id"].str, new CoreLayer(layer), (BaseLayer.TYPE)Enum.Parse(typeof(BaseLayer.TYPE), layer["layer_type"].str));
                            break;
                        case BaseLayer.RECURRENT:
                            res.AddLayer(layer["id"].str, new RecurrentLayer(layer), (BaseLayer.TYPE)Enum.Parse(typeof(BaseLayer.TYPE), layer["layer_type"].str));
                            break;
                        case BaseLayer.LSTM:
                            //res.AddLayer(layer["id"].str, new LSTMLayer(layer), (BaseLayer.TYPE)Enum.Parse(typeof(BaseLayer.TYPE), layer["layer_type"].str));
                            break;
                    }
                }

                JSONObject connections = main["connections"];

                foreach (string key in connections.keys)
                {
                    JSONObject connection = connections[key];

                    BaseLayer inLayer = connection.HasField("inlayer") ? res.Layers[connection["inlayer"].str] : null;
                    BaseLayer outLayer = connection.HasField("outlayer") ? res.Layers[connection["outlayer"].str] : null;

                    if (inLayer != null && outLayer != null) { 
                        Connection c = new Connection(key, inLayer.OutputGroup, outLayer.InputGroup, connection);

                        res.AddConnection(inLayer.Id, outLayer.Id, c);
                    }
                }
            }

            return res;
        }
        
    }
}

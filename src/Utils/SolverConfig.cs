using Newtonsoft.Json;
using System.Collections;

public class SolverConfig
{
    public int epochs;
    public float epsilon;
    public float learning_rate;
    public int hidden_layer;
    public int memory_size;
    public int batch_size;
    public int async_update;
    public int async_learners;
    public int qtupdate_size;
    public int level;
    public int seed;
    public string filename;

    private static SolverConfig _instance = null;

    public static SolverConfig GetInstance()
    {
        if (_instance == null)
        {
            _instance = new SolverConfig();
        }

        return _instance;
    }

    public void Save()
    {
        string data = JsonConvert.SerializeObject(_instance);
        System.IO.File.WriteAllText(@".\config.json", data);
    }

    public void Load(string p_filename)
    {
        string data = System.IO.File.ReadAllText(p_filename);
        _instance = JsonConvert.DeserializeObject<SolverConfig>(data);
    }

}

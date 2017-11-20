using System;
using System.IO;

public class Logger
{
    private StreamWriter _file = null;

    public Logger()
    {

    }

    public void Init(string p_filename)
    {
        _file = new StreamWriter(p_filename, false);
        Log(DateTime.Now.ToLongDateString() + " " + DateTime.Now.ToLongTimeString());
    }

    public void Log(string p_line)
    {
        if (_file != null) _file.WriteLine(p_line);
    }

    public void Close()
    {
        if (_file != null)
        {
            Log(DateTime.Now.ToLongDateString() + " " + DateTime.Now.ToLongTimeString());
            _file.Close();
        }
    }
}
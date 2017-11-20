using FLAB;
using System;
using System.Collections.Generic;

public class StateEncoder
{
    private InputInterface _inputInterface;
    Dictionary<string, int> _itemProjection;
    Dictionary<string, int> _cellProjection;

    public StateEncoder(InputInterface p_inputInterface)
    {
        _inputInterface = p_inputInterface;
        _itemProjection = new Dictionary<string, int>();
        _cellProjection = new Dictionary<string, int>();
    }

    public void PrepareEncoderProjection(Match3LevelConfig p_config)
    {
        int cellId = 0;
        int itemId = 0;

        _itemProjection = new Dictionary<string, int>();
        _cellProjection = new Dictionary<string, int>();

        foreach (Match3LevelConfig.Match3LevelCell cell in p_config.cells)
        {
            if (!_cellProjection.ContainsKey(cell.cellTypeID))
            {
                _cellProjection.Add(cell.cellTypeID, cellId);
                cellId++;
            }

            if (!_itemProjection.ContainsKey(cell.itemID))
            {
                _itemProjection.Add(cell.itemID, itemId);
                itemId++;
            }
        }

        foreach (Match3LevelConfig.Match3LevelDrop drop in p_config.drops)
        {
            if (!_itemProjection.ContainsKey(drop.itemID))
            {
                _itemProjection.Add(drop.itemID, itemId);
                itemId++;
            }
        }

        foreach (Match3LevelConfig.Match3LevelSpawn spawn in p_config.spawns)
        {
            if (!_itemProjection.ContainsKey(spawn.itemID))
            {
                _itemProjection.Add(spawn.itemID, itemId);
                itemId++;
            }
        }

        foreach (ItemType item in _inputInterface.gridDefinition.ItemTypes)
        {
            if (!_itemProjection.ContainsKey(item.Id) && item.Type == "SPECIAL")
            {
                _itemProjection.Add(item.Id, itemId);
                itemId++;
            }
        }
    }

    public int GetActionDim()
    {
        return 81 + 1;
    }

    public int GetStateDim()
    {
        return ((int)Math.Ceiling(Math.Log(_cellProjection.Count, 2)) + (int)Math.Ceiling(Math.Log(_itemProjection.Count, 2))) * 81 + 2;
    }

    public Vector EncodeState(List<GridDto> p_data, int p_powerupScore, int p_dim, bool p_boost)
    {
        List<float> tmp = new List<float>();

        foreach (GridDto gp in p_data)
        {
            tmp.AddRange(BinaryEncode(gp));
        }

        tmp.Add((float)p_powerupScore / _inputInterface.gridDefinition.BoostPoints);
        tmp.Add(p_boost ? 1f : 0f);

        Vector result = Vector.Build(p_dim, tmp);

        return result;
    }

    private float[] OneHotEncode(GridDto p_cell)
    {
        float[] result = new float[_cellProjection.Count + _itemProjection.Count];

        int c_index = _cellProjection[p_cell.idCell];
        int i_index = _cellProjection.Count + _itemProjection[p_cell.idItem];

        result[c_index] = 1;
        result[i_index] = 1;

        return result;
    }

    private float[] BinaryEncode(GridDto p_cell)
    {
        List<float> tmp = new List<float>();

        float[] c_code = DecToBin(_cellProjection[p_cell.idCell], (int)Math.Ceiling(Math.Log(_cellProjection.Count, 2)));
        float[] i_code = DecToBin(_itemProjection[p_cell.idItem], (int)Math.Ceiling(Math.Log(_itemProjection.Count, 2)));

        tmp.AddRange(c_code);
        tmp.AddRange(i_code);

        return tmp.ToArray();
    }

    private float[] DecToBin(int p_int, int p_size)
    {
        float[] result = new float[p_size];
        int i = 0;

        while (p_int > 0)
        {
            result[i] = p_int % 2;
            p_int /= 2;
            i++;
        }

        return result;
    }

}
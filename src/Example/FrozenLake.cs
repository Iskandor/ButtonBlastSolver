using System.Collections;

public class FrozenLake
{
    public const int DIM_X = 4;
    public const int DIM_Y = 3;

    private int[,] _lake;
    private int[,] _state;
    private int _x, _y;

    public FrozenLake()
    {
        /*
        _lake = new int[DIM_Y, DIM_X] { {0,0,0,0,0,0,0,0,2},
                                        {0,1,0,0,0,0,0,0,3},
                                        {0,0,0,0,0,0,0,0,0},
                                        {0,0,0,0,0,0,0,0,0},
                                        {0,0,0,0,0,0,0,0,0},
                                        {0,0,0,0,0,0,0,0,0},
                                        {0,0,0,0,0,0,0,0,0},
                                        {0,0,0,0,0,0,0,0,0},
                                        {0,0,0,0,0,0,0,0,0}};
                                        */
        _lake = new int[DIM_Y, DIM_X] 
        {
            {0,0,0,2},
            {0,1,0,3},
            {0,0,0,0}
        };
        _state = new int[DIM_Y, DIM_X];        
    }

    public void Reset()
    {
        _x = 0;
        _y = DIM_Y-1;
    }

    public void DoAction(int p_action)
    {
        int nx = _x;
        int ny = _y;

        switch(p_action)
        {
            case 0:
                ny -= 1;
                break;
            case 1:
                nx += 1;
                break;
            case 2:
                ny += 1;
                break;
            case 3:
                nx -= 1;
                break;
        }

        if (IsValid(nx, ny))
        {
            _x = nx;
            _y = ny;
        }
    }

    public int IsFinished()
    {
        int result = 0;

        if (_lake[_y, _x] == 2) result = 1;
        if (_lake[_y, _x] == 3) result = -1;

        return result;
    }

    public float GetReward()
    {
        float reward = 0f;

        if (_lake[_y, _x] == 2)
        {
            reward = 1;
        }
        if (_lake[_y, _x] == 3)
        {
            reward = -1;
        }

        return reward;
    }

    public int[,] GetState()
    {
        for(int i = 0; i < DIM_Y; i++)
        {
            for(int j = 0; j < DIM_X; j++)
            {
                _state[i, j] = _lake[i, j];
            }
        }        
        _state[_y, _x] = 4;

        return _state;
    }

    private bool IsValid(int p_x, int p_y)
    {
        return (p_x >= 0 && p_x < DIM_X && p_y >= 0 && p_y < DIM_Y && _lake[p_y, p_x] != 1);
    }
}

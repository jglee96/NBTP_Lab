function [Layer,newstate_idx] = DoActLayer(Layer,state_idx,action_idx)
Ngrid = length(Layer);
switch fix(state_idx/Ngrid)
    case 0
        switch action_idx
            case 1
                newstate_idx = state_idx-1;
            case 2
                newstate_idx = state_idx+Ngrid-1;
            case 3
                newstate_idx = state_idx+Ngrid+1;
            case 4
                newstate_idx = state_idx+1;
        end
        
        if newstate_idx-Ngrid > 0
            Layer(newstate_idx-Ngrid) = 1;
        else
            Layer(newstate_idx) = 0;
        end
    case 1
        switch action_idx
            case 1
                newstate_idx = state_idx-1;
            case 2
                newstate_idx = state_idx-Ngrid-1;
            case 3
                newstate_idx = state_idx-Ngrid+1;
            case 4
                newstate_idx = state_idx+1;
        end
        if newstate_idx-Ngrid > 0
            Layer(newstate_idx-Ngrid) = 1;
        else
            Layer(newstate_idx) = 0;
        end
end
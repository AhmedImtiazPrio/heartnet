function [idx_states,last_idx] = get_states_python(assigned_states)
    indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

    if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
        switch assigned_states(1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=4;
        end
    else
        switch assigned_states(indx(1)+1)
            case 4
                K=1;
            case 3
                K=2;
            case 2
                K=3;
            case 1
                K=0;
        end
        K=K+1;
    end

    indx2                = indx(K:end); % K controls the starting cycle
                                        % of the segment. Starting cycle
                                        % is always kept 1 through the 
                                        % switch cases (!)
                                        
    rem                  = mod(length(indx2),4);
    last_idx_             = length(indx2)-rem+1;
    try
        last_idx             = indx2(last_idx_);
    catch ME
        disp(ME.message)    
        last_idx             = length(assigned_states);
    end
    indx2(last_idx_:end) = []; % clipping the partial segments in the end
    idx_states           = reshape(indx2,4,length(indx2)/4)'; % idx_states 
                            % reshaped into a no.segments X 4 sized matrix
                            % containing state indices
end
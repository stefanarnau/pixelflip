function apes = apes_t(t, df_error)
    
    %
    % This function calculates the effect size based on t-values
    % as adjusted partial eta squared (Mordkoff, 2019).
    %
    %
    % Inputs:
    % t        = The t-value
    % df_error = The corresponding degrees of freedom
    %
    % Output:
    %
    % apes     = adjusted partial eta squared
    %

    x = t^2 / (t^2 + df_error);
    apes = x - (1 - x) * (1 / df_error);
    
end
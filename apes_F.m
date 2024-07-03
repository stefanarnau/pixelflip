function apes = apes_F(F, df_effect, df_error)

    %
    % This function calculates the effect size based on F values as 
    % adjusted partial eta squared (Mordkoff, 2019).
    %
    % Inputs:
    % F         = The F value
    % df_effect = degrees of freedom of conditions
    % df_error  = degrees of freedom of sample
    %
    % Output:
    % apes      = The effect size as adjusted partial eta squared
    %

    x = (F * df_effect) / (F * df_effect + df_error);
    apes = x - (1 - x) * (df_effect / df_error);

end
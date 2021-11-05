function [Z,S,L,obj] = FLLRRWDL1(X,Z_ini,SD,lambda1,lambda2,lambda3,max_iter,Ctg,miu,rho)
    % The code is written by Jie Wen, 
    % if you have any problems, please don't hesitate to contact me: wenjie@hrbeu.edu.cn 
    % If you find the code is useful, please cite the following reference:
    % J. Wen, B. Zhang, Y. Xu, J. Yang, and N. Han, 
    % Adaptive Weighted Nonnegative Low-Rank Representation, 
    % Pattern Recognition, 2018.
    
    max_miu = 1e8;
    tol  = 1e-6;
    tol2 = 1e-2;
    
    S = ones(size(X));
    L = zeros(size( X,1),size(X,1));
    V = L;
    C1 = zeros(size(X));
    C2 = zeros(size(Z_ini));
    C3 = zeros(size(S,1),1);
    C4 = zeros(size(L));
    distX = L2_distance_1(X,X);
    D = lambda3*SD.*distX;
    % D = lambda3*distX;
    one1 = ones(size(S,2),1);
    one2 = ones(size(S,1),1);
    for iter = 1:max_iter
        if iter == 1
            Z = Z_ini;
            U = Z_ini;
            E = X-X*Z-L*X;
        end
        Z_old = Z;
        U_old = U; 
        E_old = E;
        S_old = S;
        L_old = L;
        V_old = V;
    
        % ------------ S ------------- %不用改
        S_linshi = -(abs(E))/lambda1;
    % %     S_linshi =(miu*one2*one1'-C3*one1'-abs(E))*pinv(miu*one1*one1'+lambda1*eye(size(S,2),size(S,2)));
        S = zeros(size(S_linshi));
        for ii = 1:size(E,2)
            S(:,ii) = EProjSimplex(S_linshi(:,ii));
        end
    %% L1-norm S
    %     S =(miu*one2*one1'-C3*one1'-abs(E))*pinv(miu*one1*one1'+lambda1*eye(size(S,2),size(S,2)));
    %     S = max(0,S);
        % --------- E -------- %已改
    %     G = X-X*Z -L*X +C1/miu;
    %     E = (miu*G)./(miu+2*S);
        E = updataE(E,S,X*Z+L*X,X,C1,miu);
        % -------- Z ------------ %已改
        M1 = X-E-L*X+C1/miu;
        M2 = U-C2/miu;
        Z = Ctg*(X'*M1+M2-D/miu);
        Z = Z - diag(diag(Z));
        for ii = 1:size(Z,2)
            idx = 1:size(Z,2);
    %         place = SD(ii,:);
    %         place = find(place >1);
    %         Z(ii,place) = 0;
            idx(ii) = [];
            Z(ii,idx) = EProjSimplex_new(Z(ii,idx));
        end
        % ------------ U ------------ %
        %tempU = Z+C2/miu;
        %[AU,SU,VU] = svd(tempU,'econ');
        %AU(isnan(AU)) = 0;
        %VU(isnan(VU)) = 0;
        %SU(isnan(SU)) = 0;
        %SU = diag(SU);    
        %SVP = length(find(SU>lambda2/miu));
        %if SVP >= 1
        %    SU = SU(1:SVP)-lambda2/miu;
        %else
        %    SVP = 1;
        %    SU = 0;
        %end
        %U = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';
        tempU = Z+C2/miu;
        U= miu*tempU/(lambda2+miu);

        %% ---------V
        %tempU = L + C4/miu;
        %[AU,SU,VU] = svd(tempU,'econ');
        %AU(isnan(AU)) = 0;
        %VU(isnan(VU)) = 0;
        %SU(isnan(SU)) = 0;
        %SU = diag(SU);    
        %SVP = length(find(SU>lambda2/miu));
        %if SVP >= 1
        %    SU = SU(1:SVP)-lambda2/miu;
        %else
        %    SVP = 1;
        %    SU = 0;
        %end
        %V = AU(:,1:SVP)*diag(SU)*VU(:,1:SVP)';
        tempU = L + C4/miu;
        V= miu*tempU/(lambda2+miu);
        %% L
        L = (V+C1/miu*X'-C4/miu + X*X' - E*X' - X*Z*X')*pinv(X*X' + eye(size(X,1),size(X,1)));
     
        % ------ C1 C2 miu ---------- %
        L1 = X-X*Z - L*X -E;
        L2 = Z-U;
        L3 = S*one1 - one2;
        L4 = L - V;
        C1 = C1+miu*L1;
        C2 = C2+miu*L2;
        C3 = C3+miu*L3;
        C4 = C4+miu*L4;
        
        LL1 = norm(Z-Z_old,'fro');
        LL2 = norm(U-U_old,'fro');
        LL3 = norm(E-E_old,'fro');
        LL4 = norm(S-S_old,'fro');
        LL5 = norm(L-L_old,'fro');
        LL6 = norm(V-V_old,'fro');
        SLSL = [LL1,LL2,LL3,LL4,LL5,LL6];
        SLSL = max(SLSL)/norm(X,'fro');
    %     if miu*SLSL < tol2
            miu = min(rho*miu,max_miu); 
    %     end
        stopC = (norm(L1,'fro')+norm(L2,'fro')+norm(L3,'fro')+norm(L4,'fro'))/norm(X,'fro');
        if stopC < tol
            iter
            break;
        end
        obj(iter) = stopC;
    end
    end
    function E = updataE(E,S,XZ,X,C1,miu)%这个计算速度还可以优化
        place1 = find(E>=0);
        place2 = find(E<0);
        E(place1) = -S(place1)/miu-XZ(place1)+X(place1)+C1(place1)/miu;
        E(place2) =  S(place2)/miu-XZ(place2)+X(place2)+C1(place2)/miu;
    end
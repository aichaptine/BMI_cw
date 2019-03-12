function out_updated = convert_to_RA(out)

out_updated = zeros(size(out));
a = zeros(1,size(out,2));
m = 0;

    for i=1:7
        for j= i+1:8
            m = m+1;
            a(out(m,:)==1) = i;
            a(out(m,:)==0) = j;
            out_updated(m,:) = a;
        end
    end
    
end

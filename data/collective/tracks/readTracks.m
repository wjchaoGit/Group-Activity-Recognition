function tracks = readTracks(filename)
fp = fopen(filename, 'r');

tline = fgetl(fp);
nframe = sscanf(tline, 'Total frames %d');

tline = fgetl(fp);
ntargets = sscanf(tline, 'Number of Targets %d');


for n = 1:ntargets
    track = struct('id', n, 'ti', 0, 'te', 0, 'bbs', [], 'locs', []);
    
    tline = fgetl(fp);
    temp = sscanf(tline, 'Target %d (frames from %d to %d)');
    track.id = temp(1);     track.ti = temp(2);     track.te = temp(3);
    
    len = temp(3) - temp(2) + 1;
    tline = fgetl(fp); % dummy line
    for t = 1:len
        tline = fgetl(fp);
        temp = sscanf(tline, '%d\t%d\t%d\t%d\t%d');
        track.bbs(:, t) = temp(2:5);
    end
    
    tline = fgetl(fp); % dummy line
    for t = 1:len
        tline = fgetl(fp);
        temp = sscanf(tline, '%d\t%f\t%f\t%f\t%f');
        track.locs(:, t) = temp(2:5);
    end
    
    tracks(n) = track;
end

fclose(fp);

end
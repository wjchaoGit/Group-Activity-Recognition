function showTracks(imdir, tracks)

imfiles = dir([imdir '*.jpg']);

for i = 1:length(imfiles)
    imshow([imdir imfiles(i).name]);
    
    drawTracks(tracks, i);
    
    drawnow;
end

end



function drawTracks(tracks, frame)

cmap = colormap;

for i = 1:length(tracks)
    if ((tracks(i).ti <= frame) & ...
        (tracks(i).te >= frame))
        idx = frame - tracks(i).ti + 1;

        col = cmap(mod(i*10, 64) + 1, :);
        rectangle('Position', tracks(i).bbs(:, idx), 'EdgeColor', col, 'LineWidth', 3);
    end
end

end
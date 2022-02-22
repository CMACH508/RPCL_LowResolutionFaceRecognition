clear,close all
% load the corresponding probe&gallery ids
load('gallery_match_img_ID_pairs.mat');
load('probe_img_ID_pairs.mat');

%% load your features, please name your feature matrix as
% "gallery_feature_map" & "probe_feature_map" & "distractor_feature_map",
% where each feature map's dimension should be
% [image_number]_by_[feature_dimension]

feature_folder = '';
fprintf('%s\n', feature_folder)
load([feature_folder 'gallery.mat']);
load([feature_folder 'probe.mat']);
load([feature_folder 'distractor.mat']);

%% gallery/query construction
% building the all the gallery set by combining the match gallery and distractor features
gallery_feature_map = [gallery_feature_map; distractor_feature_map];
% buidling the corresponding gallery ids, all the distractors are labeled as
% -100, while mated gallery are labeled by corresponding ids (to-be-matched to
% probe)
distractor_ids = -100 * ones(size(distractor_feature_map,1),1);
gallery_ids = [gallery_ids; distractor_ids];

dist = pdist2(gallery_feature_map,probe_feature_map, 'euclidean');
CMC = [];
ap = [];
for p = 1:size(probe_feature_map,1)
    probe_id = probe_ids(p,1);
    good_index = find(gallery_ids == probe_id);
    score = dist(:, p);
    [~, index] = sort(score, 'ascend');  % single query
    [ap(p), CMC(p, :)] = compute_AP(good_index, index);% compute AP for single query
end
CMC = mean(CMC);
%% print result
fprintf('mAP = %f, r1 precision = %f, r5 precision = %f, r10 precision = %f, r20 precision = %f, r50 precision = %f\r\n', mean(ap), CMC(1),CMC(5),CMC(10),CMC(20),CMC(50));

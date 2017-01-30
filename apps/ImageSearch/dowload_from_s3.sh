#!/usr/bin/env bash

aws s3 cp s3://image-search-zappos/projections_all.npy .
aws s3 cp s3://image-search-zappos/NN_order.npy .
aws s3 cp s3://image-search-zappos/features_d1000.npy .
aws s3 cp s3://image-search-zappos/hash_object.npy .
aws s3 cp s3://image-search-zappos/hash_object.pkl .
aws s3 cp s3://image-search-zappos/hash_object_nonquad.npy .
aws s3 cp s3://image-search-zappos/lsh_index_array.npy .
aws s3 cp s3://image-search-zappos/projections_nonquad.npy .
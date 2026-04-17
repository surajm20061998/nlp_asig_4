import torch
from load_data import load_t5_data
from t5_utils import initialize_model
import argparse

def test_data_loading():
    print("\n" + "="*60)
    print("Testing Data Loading...")
    print("="*60)
    
    try:
        train_loader, dev_loader, test_loader = load_t5_data(batch_size=4, test_batch_size=4)
        
        print("Train loader created: {} batches".format(len(train_loader)))
        print("Dev loader created: {} batches".format(len(dev_loader)))
        print("Test loader created: {} batches".format(len(test_loader)))
        
        encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs = next(iter(train_loader))
        print("\nTrain batch shapes:")
        print("  Encoder input: {}".format(encoder_ids.shape))
        print("  Encoder mask: {}".format(encoder_mask.shape))
        print("  Decoder input: {}".format(decoder_input_ids.shape))
        print("  Decoder target: {}".format(decoder_target_ids.shape))
        
        print("\nData loading test PASSED!")
        return True
        
    except Exception as e:
        print("\nData loading test FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    print("\n" + "="*60)
    print("Testing Model Initialization...")
    print("="*60)
    
    try:
        args = argparse.Namespace(finetune=True)
        model = initialize_model(args)
        print("Fine-tuning model initialized")
        print("  Model type: {}".format(type(model).__name__))
        
        args = argparse.Namespace(finetune=False)
        model = initialize_model(args)
        print("From-scratch model initialized")
        
        print("\nModel initialization test PASSED!")
        return True
        
    except Exception as e:
        print("\nModel initialization test FAILED: {}".format(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("TESTING PART 2 IMPLEMENTATION")
    print("="*60)
    
    results = []
    results.append(("Data Loading", test_data_loading()))
    results.append(("Model Initialization", test_model_initialization()))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print("{}: {}".format(test_name, status))
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now:")
        print("1. Run compute_data_statistics.py to get Q4 statistics")
        print("2. Run train_t5.py to train the baseline model")
        print("="*60)
    else:
        print("\nSOME TESTS FAILED!")

if __name__ == "__main__":
    main()

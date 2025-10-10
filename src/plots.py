"""
Code to create visualizations
"""

from pathlib import Path
import torch
import numpy as np
import json
from src.config import Config
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):

#     pass


# if __name__ == "__main__":
#     main()


class ModelAnalyzer:
    def __init__(self, output_dir: str = "reports/model_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_loss_plot(self,
                        train_losses: List[float],
                        val_losses: List[float],
                        config: Config, 
                        train_accuracies: Optional[List[float]] = None,
                        val_accuracies: Optional[List[float]] = None
                        ):

        if len(train_losses) != len(val_losses):
            print(f"Warning - different length of lists - train: {len(train_losses)}, val: {len(val_losses)}")
            min_len = min(len(train_losses), len(val_losses))
            train_losses = train_losses[:min_len]
            val_losses = val_losses[:min_len]
            if train_accuracies and val_accuracies:
                train_accuracies = train_accuracies[:min_len] 
                val_accuracies = val_accuracies[:min_len]
    
        epochs = range(1, len(train_losses) + 1)
        
        if train_accuracies is not None and val_accuracies is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        
        # Wykres strat
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=1, marker='o', markersize=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=1, marker='s', markersize=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training loss vs validation loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if train_accuracies is not None and val_accuracies is not None:
            ax2.plot(epochs, train_accuracies, 'g-', label='Training Accuracy', linewidth=1, marker='o', markersize=2)
            ax2.plot(epochs, val_accuracies, 'm-', label='Validation Accuracy', linewidth=1, marker='s', markersize=2)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training accuracy vs validation accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
    
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"loss_plot_{config.dataset_name}_{config.optimizer_config.optimizer_name}_{timestamp}.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"Loss plot written in: {self.output_dir / plot_filename}")

    def create_box_plots(self, val_loss_from_results, acc_from_results, config):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.boxplot(y=val_loss_from_results, ax=axes[0], color="skyblue",  medianprops={'color': 'red', 'linewidth': 2})
        axes[0].set_title("Boxplot Val Loss")
        sns.boxplot(y=acc_from_results, ax=axes[1], color="lightgreen",  medianprops={'color': 'red', 'linewidth': 2})
        axes[1].set_title("Boxplot Val Accuracy")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"boxplot_plot_{config.dataset_name}_{config.optimizer_config.optimizer_name}_{timestamp}.png"
        plt.savefig(self.output_dir / "figures" / plot_filename)
        plt.show()
    

    def create_violin_plots(self, val_loss_from_results, acc_from_results, config):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.violinplot(y=val_loss_from_results, ax=axes[0], color="skyblue",  inner="quartile",
                   inner_kws=dict(color="red", linewidth=2))
        
        axes[0].set_title("Violin Plot Val Loss")
        sns.violinplot(y=acc_from_results, ax=axes[1], color="lightgreen",  inner="quartile",
                   inner_kws=dict(color="red", linewidth=2))
        axes[1].set_title("Violin Plot Accuracy")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"violin_plot_{config.dataset_name}_{config.optimizer_config.optimizer_name}_{timestamp}.png"
        plt.savefig(self.output_dir / "figures" / plot_filename)
        plt.show()
    

    def create_n_seed_losses_plot(self, all_val_losses, config):
        all_val_losses = np.array(all_val_losses)

        final_losses = all_val_losses[:, -1]
        median_final = np.median(final_losses)
        closest_idx = np.argmin(np.abs(final_losses - median_final))

        plt.figure(figsize=(10, 6))
        for i, loss_curve in enumerate(all_val_losses):
            if i == closest_idx:
                plt.plot(loss_curve, color="red", linewidth=2.5, label="Median seed")
            else:
                plt.plot(loss_curve, color="gray", alpha=0.5, linewidth=1)

        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss across seeds")
        plt.legend()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"n_seed_plot_{config.dataset_name}_{config.optimizer_config.optimizer_name}_{timestamp}.png"
        plt.savefig(self.output_dir / "figures" / plot_filename)
        plt.show()
    
    def analyze_model(self, model: torch.nn.Module, config: Dict[str, Any], 
                     final_loss: float, final_accuracy: float, 
                     initialization_type: str = "default") -> Dict[str, Any]:
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "initialization_type": initialization_type,
            "final_metrics": {
                "loss": final_loss,
                "accuracy": final_accuracy
            },
            "layer_statistics": {},
            "global_statistics": {}
        }
        
        # Analiza per warstwa
        layer_stats = []
        all_weights = []
        all_biases = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                
                if 'weight' in name:
                    all_weights.extend(weights)
                    layer_stat = {
                        "layer_name": name,
                        "shape": list(param.shape),
                        "num_parameters": param.numel(),
                        "mean": float(np.mean(weights)),
                        "std": float(np.std(weights)),
                        "min": float(np.min(weights)),
                        "max": float(np.max(weights)),
                        "median": float(np.median(weights)),
                        "l1_norm": float(np.linalg.norm(weights, ord=1)),
                        "l2_norm": float(np.linalg.norm(weights, ord=2)),
                        "percentage_zero": float(np.mean(np.abs(weights) < 1e-6) * 100),
                        "percentage_negative": float(np.mean(weights < 0) * 100)
                    }
                    analysis["layer_statistics"][name] = layer_stat
                    layer_stats.append(layer_stat)
                
                elif 'bias' in name and param is not None:
                    all_biases.extend(weights)
        
        # Statystyki globalne
        if all_weights:
            analysis["global_statistics"]["weights"] = {
                "total_parameters": len(all_weights),
                "mean": float(np.mean(all_weights)),
                "std": float(np.std(all_weights)),
                "min": float(np.min(all_weights)),
                "max": float(np.max(all_weights)),
                "l1_norm": float(np.linalg.norm(all_weights, ord=1)),
                "l2_norm": float(np.linalg.norm(all_weights, ord=2)),
                "weight_range": float(np.max(all_weights) - np.min(all_weights))
            }
        
        if all_biases:
            analysis["global_statistics"]["biases"] = {
                "total_parameters": len(all_biases),
                "mean": float(np.mean(all_biases)),
                "std": float(np.std(all_biases)),
                "min": float(np.min(all_biases)),
                "max": float(np.max(all_biases))
            }
        
        self._save_analysis(analysis, initialization_type)
        #self._create_visualizations(model, analysis, initialization_type)
        
        return analysis
    
    def _save_analysis(self, analysis: Dict[str, Any], init_type: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{init_type}_{timestamp}.json"
        
        with open(self.output_dir / filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analiza zapisana do: {self.output_dir / filename}")
    
    def _create_visualizations(self, model: torch.nn.Module, analysis: Dict[str, Any], init_type: str):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Analiza wag - {init_type}', fontsize=16)
        
        all_weights = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weights = param.data.cpu().numpy().flatten()
                all_weights.extend(weights)
                layer_names.append(name)
        
        # 1. Histogram wszystkich wag
        axes[0, 0].hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Rozkład wszystkich wag')
        axes[0, 0].set_xlabel('Wartość wagi')
        axes[0, 0].set_ylabel('Częstość')
        axes[0, 0].axvline(np.mean(all_weights), color='red', linestyle='--', label=f'Średnia: {np.mean(all_weights):.4f}')
        axes[0, 0].legend()
        
        # 2. Statystyki per warstwa
        layer_means = [analysis["layer_statistics"][name]["mean"] for name in analysis["layer_statistics"]]
        layer_stds = [analysis["layer_statistics"][name]["std"] for name in analysis["layer_statistics"]]
        
        x_pos = range(len(layer_means))
        axes[0, 1].bar(x_pos, layer_means, alpha=0.7)
        axes[0, 1].set_title('Średnia wag per warstwa')
        axes[0, 1].set_xlabel('Warstwa')
        axes[0, 1].set_ylabel('Średnia wartość')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([analysis["layer_statistics"][name]["layer_name"] for name in analysis["layer_statistics"]], rotation=45)
        
        # 3. Odchylenie standardowe per warstwa
        axes[1, 0].bar(x_pos, layer_stds, alpha=0.7, color='orange')
        axes[1, 0].set_title('Odchylenie standardowe per warstwa')
        axes[1, 0].set_xlabel('Warstwa')
        axes[1, 0].set_ylabel('Std')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([analysis["layer_statistics"][name]["layer_name"] for name in analysis["layer_statistics"]], rotation=45)
        
        # 4. Min/Max per warstwa
        layer_mins = [analysis["layer_statistics"][name]["min"] for name in analysis["layer_statistics"]]
        layer_maxs = [analysis["layer_statistics"][name]["max"] for name in analysis["layer_statistics"]]
        
        axes[1, 1].plot(x_pos, layer_mins, 'o-', label='Min', color='blue')
        axes[1, 1].plot(x_pos, layer_maxs, 'o-', label='Max', color='red')
        axes[1, 1].set_title('Min/Max wag per warstwa')
        axes[1, 1].set_xlabel('Warstwa')
        axes[1, 1].set_ylabel('Wartość')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([analysis["layer_statistics"][name]["layer_name"] for name in analysis["layer_statistics"]], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"weights_analysis_{init_type}_{timestamp}.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Wizualizacja zapisana do: {self.output_dir / plot_filename}")
    

    def compare_initializations(self, analysis_files: List[str]):
        analyses = []
        
        for file_path in analysis_files:
            with open(file_path, 'r') as f:
                analysis = json.load(f)
                analyses.append(analysis)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Initializations comparsion', fontsize=16)
        
        init_types = [a.get("initialization_type", "unknown") for a in analyses]
        final_losses = [a["final_metrics"]["loss"] for a in analyses]
        final_accuracies = [a["final_metrics"]["accuracy"] for a in analyses]
       
        # 1. Porównanie finalnych metryk
        x_pos = range(len(init_types))
        axes[0, 0].bar(x_pos, final_losses, alpha=0.7)
        axes[0, 0].set_title('Final loss')
        axes[0, 0].set_ylabel('Loss')
        
        axes[0, 1].bar(x_pos, final_accuracies, alpha=0.7, color='green')
        axes[0, 1].set_title('Final Accuracy')
        axes[0, 1].set_ylabel('Accuracy (%)')
    
        
        # 2. Porównanie statystyk wag
        global_means = [a["global_statistics"]["weights"]["mean"] for a in analyses]
        global_stds = [a["global_statistics"]["weights"]["std"] for a in analyses]
        
        axes[1, 0].bar(x_pos, global_means, alpha=0.7, color='orange')
        axes[1, 0].set_title('Global mean of weights')
        axes[1, 0].set_ylabel('Mean')
        
        axes[1, 1].bar(x_pos, global_stds, alpha=0.7, color='purple')
        axes[1, 1].set_title('Standard deviation of weights')
        axes[1, 1].set_ylabel('Std')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = f"initialization_comparison_{timestamp}.png"
        plt.savefig(self.output_dir / comparison_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Porównanie zapisane do: {self.output_dir / comparison_filename}")


        layer_names = list(analyses[0]["layer_statistics"].keys())
        num_layers = len(layer_names)
        
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))
        fig.suptitle('Porównanie maksymalnych wag per warstwa', fontsize=16)
        
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, layer_name in enumerate(layer_names):
            max_weights = []
            
            for analysis in analyses:
                if layer_name in analysis["layer_statistics"]:
                    max_weights.append(analysis["layer_statistics"][layer_name]["max"])
                else:
                    max_weights.append(0)
            
            x_pos = range(len(init_types))
            bars = axes[layer_idx].bar(x_pos, max_weights, alpha=0.7, 
                                    color=plt.cm.tab10(layer_idx))
            
            axes[layer_idx].set_title(f'{layer_name.split(".")[-2]}.{layer_name.split(".")[-1]}')
            axes[layer_idx].set_ylabel('Max waga')
            axes[layer_idx].grid(True, alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars, max_weights)):
                axes[layer_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"max_weights_comparison_{timestamp}.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 6))
        fig.suptitle('Porównanie minimalnych wag per warstwa', fontsize=16)
        
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, layer_name in enumerate(layer_names):
            max_weights = []
            
            for analysis in analyses:
                if layer_name in analysis["layer_statistics"]:
                    max_weights.append(analysis["layer_statistics"][layer_name]["min"])
                else:
                    max_weights.append(0)
            
            x_pos = range(len(init_types))
            bars = axes[layer_idx].bar(x_pos, max_weights, alpha=0.7, 
                                    color=plt.cm.tab10(layer_idx))
            
            axes[layer_idx].set_title(f'{layer_name.split(".")[-2]}.{layer_name.split(".")[-1]}')
            axes[layer_idx].set_ylabel('Min waga')
            axes[layer_idx].grid(True, alpha=0.3)
            
            for i, (bar, value) in enumerate(zip(bars, max_weights)):
                axes[layer_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"min_weights_comparison_{timestamp}.png"
        plt.savefig(self.output_dir / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
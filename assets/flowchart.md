## 🔄 Model Architecture

```mermaid
graph TD
    subgraph Data Layer["Data Layer"]
        style Data Layer fill:#e6f3ff,stroke:#4d94ff
        A[Excel Files] --> B[Data Processing]
        B --> C[Monthly Time Series]
        style A fill:#cce6ff,stroke:#4d94ff
        style B fill:#cce6ff,stroke:#4d94ff
        style C fill:#cce6ff,stroke:#4d94ff
    end

    subgraph Training Pipeline["Training Pipeline"]
        style Training Pipeline fill:#fff0f3,stroke:#ff4d6d
        C --> D[Data Validation]
        D --> E[Cross-Validation]
        E --> F[Hyperparameter Tuning]
        F --> G[Model Training]
        G --> H[Model Evaluation]
        H --> I[Save Model & Metrics]
        H -.-> E
        style D fill:#ffe6eb,stroke:#ff4d6d
        style E fill:#ffe6eb,stroke:#ff4d6d
        style F fill:#ffe6eb,stroke:#ff4d6d
        style G fill:#ffe6eb,stroke:#ff4d6d
        style H fill:#ffe6eb,stroke:#ff4d6d
        style I fill:#ffe6eb,stroke:#ff4d6d
    end

    subgraph Forecasting Pipeline["Forecasting Pipeline"]
        style Forecasting Pipeline fill:#f0fff4,stroke:#4dff88
        J[Load Model] --> K[Generate Forecast]
        K --> L[Post-Processing]
        L --> M[Output Results]
        style J fill:#e6ffe6,stroke:#4dff88
        style K fill:#e6ffe6,stroke:#4dff88
        style L fill:#e6ffe6,stroke:#4dff88
        style M fill:#e6ffe6,stroke:#4dff88
    end

    subgraph Utility Functions["Utility Functions"]
        style Utility Functions fill:#fff8e6,stroke:#ffb84d
        N[Data Validation]
        O[CV Parameters]
        P[Visualization]
        Q[Metrics Calculation]
        style N fill:#fff2cc,stroke:#ffb84d
        style O fill:#fff2cc,stroke:#ffb84d
        style P fill:#fff2cc,stroke:#ffb84d
        style Q fill:#fff2cc,stroke:#ffb84d
    end

    B -.-> N
    E -.-> O
    H -.-> P
    H -.-> Q
    K -.-> P
    I --> J
```

The diagram above illustrates the main components and data flow of the H2S Demand Forecasting System:

1. **Data Layer**: Handles data ingestion and preprocessing
2. **Training Pipeline**: Manages model training and evaluation
3. **Forecasting Pipeline**: Handles prediction generation
4. **Utility Functions**: Provides shared functionality across components

Each component is designed to be modular and maintainable, with clear separation of concerns and robust error handli.

